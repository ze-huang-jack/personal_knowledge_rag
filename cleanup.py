import re
import hashlib
import random
import unicodedata
from typing import List, Set, Tuple, Dict
from collections import defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# MinHash parameters — generated once at import time
# ---------------------------------------------------------------------------
_MERSENNE_PRIME = (1 << 61) - 1
_NUM_HASHES = 128
_random = random.Random(42)  # deterministic seed
_HASH_PARAMS = [
    (_random.randint(1, _MERSENNE_PRIME - 1), _random.randint(0, _MERSENNE_PRIME - 1))
    for _ in range(_NUM_HASHES)
]


@dataclass
class CleanedDocument:
    text: str
    metadata: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def normalize_text(text: str) -> str:
    """Normalize Unicode, collapse whitespace, strip garbage characters."""
    text = unicodedata.normalize("NFKC", text)
    # Replace zero-width and control characters (keep newlines)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏  ﻿]", "", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    # Remove lines that are purely non-alphanumeric symbols
    text = "\n".join(
        line for line in text.splitlines()
        if not re.match(r"^[^\w\s]{5,}$", line)
    )
    return text.strip()


def detect_repeated_lines(chunks: List[str], threshold: float = 0.6) -> Set[str]:
    """Detect lines that appear across many chunks (headers, footers, boilerplate).

    Lines appearing in >threshold fraction of chunks are flagged as noise.
    """
    if len(chunks) < 3:
        return set()

    from collections import Counter
    line_counts: Counter = Counter()
    for chunk in chunks:
        unique_lines = set(chunk.splitlines())
        for line in unique_lines:
            line = line.strip()
            if len(line) > 5:  # ignore very short lines
                line_counts[line] += 1

    min_occurrences = max(2, int(len(chunks) * threshold))
    return {line for line, count in line_counts.items() if count >= min_occurrences}


def filter_noise_lines(text: str, noise_patterns: Set[str]) -> str:
    """Remove lines matching known noise patterns."""
    lines = text.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if stripped in noise_patterns:
            continue
        # Common boilerplate patterns
        if re.match(r"^(第\s*\d+\s*页|Page\s+\d+|\d+\s*/\s*\d+)$", stripped):
            continue
        if re.match(r"^(版权所有|Copyright\s*[©©]\s*\d{4}|All Rights Reserved)", stripped):
            continue
        if re.match(r"^(https?://|www\.)", stripped) and len(stripped) < 50:
            continue
        kept.append(line)
    return "\n".join(kept)


def _shingle(text: str, n: int = 5) -> Set[int]:
    """Generate character n-gram hashes for MinHash."""
    text = text.lower()
    shingles = set()
    for i in range(len(text) - n + 1):
        h = hashlib.blake2b(text[i:i + n].encode(), digest_size=8).digest()
        shingles.add(int.from_bytes(h, "big"))
    return shingles


def _minhash_signature(shingles: Set[int]) -> List[int]:
    """Generate MinHash signature using universal hashing: h(x) = (a*x + b) % p.

    O(num_hashes * num_shingles) with simple integer math — no crypto hashing.
    """
    sig = [float("inf")] * _NUM_HASHES
    for s in shingles:
        for i, (a, b) in enumerate(_HASH_PARAMS):
            h = (a * s + b) % _MERSENNE_PRIME
            if h < sig[i]:
                sig[i] = h
    return [int(v) for v in sig]


def _estimate_jaccard(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def deduplicate_chunks(chunks: List[str], threshold: float = 0.85) -> Tuple[List[str], List[int]]:
    """Remove near-duplicate chunks using MinHash + LSH banding.

    LSH reduces pairwise checks from O(n²) to candidate pairs only.
    Returns (unique_chunks, removed_indices).
    """
    if len(chunks) <= 1:
        return chunks, []

    signatures = [_minhash_signature(_shingle(c)) for c in chunks]

    # Partition signatures into bands for LSH
    # 128 hashes / 16 bands = 8 rows per band
    bands = 16
    rows_per_band = _NUM_HASHES // bands

    # For each band, bucket indices by band-hash
    # Then build candidate pairs: indices that share at least one bucket
    seen_pairs: Set[Tuple[int, int]] = set()
    candidate_neighbours: Dict[int, Set[int]] = defaultdict(set)

    for band_idx in range(bands):
        start = band_idx * rows_per_band
        end = start + rows_per_band
        bucket: Dict[int, List[int]] = defaultdict(list)

        for idx, sig in enumerate(signatures):
            band_hash = hash(tuple(sig[start:end]))
            bucket[band_hash].append(idx)

        for idxs in bucket.values():
            if len(idxs) <= 1:
                continue
            # Create candidate pairs within this bucket
            for p in range(len(idxs)):
                for q in range(p + 1, len(idxs)):
                    a, b = idxs[p], idxs[q]
                    if a > b:
                        a, b = b, a
                    pair = (a, b)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        candidate_neighbours[b].add(a)
    # end LSH bucketing

    # Dedup: first-occurrence-wins, only check LSH candidates
    kept: List[int] = []
    removed: List[int] = []

    for i in range(len(chunks)):
        is_duplicate = False
        candidates = candidate_neighbours.get(i, set())
        for j in kept:
            if j not in candidates and i not in candidate_neighbours.get(j, set()):
                continue  # not an LSH candidate — skip cheaply
            if _estimate_jaccard(signatures[i], signatures[j]) > threshold:
                is_duplicate = True
                break
        if is_duplicate:
            removed.append(i)
        else:
            kept.append(i)

    return [chunks[i] for i in kept], removed


def extract_metadata(raw_text: str, filename: str = "") -> dict:
    """Extract document-level metadata: title candidate, language hints, section count."""
    metadata = {"filename": filename, "char_count": len(raw_text)}

    lines = raw_text.splitlines()
    # First non-empty line is likely the title
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) > 3:
            metadata["title_candidate"] = stripped[:120]
            break

    # Count section-like lines
    heading_pattern = re.compile(r"^(#{1,6}\s|第[一二三四五六七八九十\d]+[章节]|\d+\.\s*\w)")
    metadata["section_count"] = sum(1 for line in lines if heading_pattern.match(line))

    return metadata


def clean_document(raw_text: str, filename: str = "") -> CleanedDocument:
    """Full cleaning pipeline: normalize → filter noise → dedup metadata.

    Call this between document loading and chunking.
    """
    metadata = extract_metadata(raw_text, filename)
    text = normalize_text(raw_text)

    # Rough pre-chunk to detect repeated headers/footers
    # Split on double newlines to approximate paragraphs
    rough_chunks = [p for p in text.split("\n\n") if len(p.strip()) > 20]
    noise_lines = detect_repeated_lines(rough_chunks)
    text = filter_noise_lines(text, noise_lines)

    removed_chars = len(raw_text) - len(text)

    return CleanedDocument(
        text=text,
        metadata=metadata,
        stats={
            "original_chars": len(raw_text),
            "cleaned_chars": len(text),
            "noise_removed_pct": round(removed_chars / max(len(raw_text), 1) * 100, 1),
            "noise_lines_detected": len(noise_lines),
        },
    )
