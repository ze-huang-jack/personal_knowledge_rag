import os
import shutil
import uuid
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cleanup import clean_document, deduplicate_chunks

load_dotenv()

DATA_DIR = Path("./data")
CHROMA_DIR = Path("./chroma_db")
COLLECTION_NAME = "knowledge_base"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def _get_llm():
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek/deepseek-v4-flash"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
        timeout=60,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def _get_embeddings():
    return OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        timeout=60,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def _get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=_get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------

def parse_file(file_path: str, original_filename: str) -> str:
    """Parse a file into clean text.

    PDF parsing has three fallback layers:
    1. markitdown  (best quality — preserves tables/headings)
    2. pymupdf     (fast, reliable, already installed)
    3. pypdf       (pure Python, no binary deps)
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        errors = []

        # Layer 1: markitdown
        try:
            from markitdown import MarkItDown
            md = MarkItDown()
            result = md.convert(file_path)
            return result.text_content
        except Exception as e:
            errors.append(f"markitdown: {e}")

        # Layer 2: pymupdf (fitz)
        try:
            import fitz
            doc = fitz.open(file_path)
            pages = [page.get_text() for page in doc]
            doc.close()
            return "\n\n".join(pages)
        except Exception as e:
            errors.append(f"pymupdf: {e}")

        # Layer 3: pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except Exception as e:
            errors.append(f"pypdf: {e}")

        raise RuntimeError(
            f"Failed to parse PDF '{original_filename}': {'; '.join(errors)}"
        )

    elif ext in (".md", ".markdown", ".txt", ".text"):
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")

    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_document(text: str, metadata: dict) -> List[Document]:
    """Split cleaned text into overlapping chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    docs = splitter.create_documents(
        texts=[text],
        metadatas=[metadata],
    )
    return docs


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_file(file_id: str) -> dict:
    """Full ingestion pipeline: parse → clean → chunk → dedup → embed → store.

    Returns stats about the ingestion for the API response.
    """
    file_dir = DATA_DIR / file_id
    files = list(file_dir.glob("*"))
    if not files:
        raise FileNotFoundError(f"No file found for id {file_id}")

    target = files[0]
    original_filename = target.name

    raw_text = parse_file(str(target), original_filename)
    cleaned = clean_document(raw_text, original_filename)

    metadata = {
        **cleaned.metadata,
        "file_id": file_id,
        "file_name": original_filename,
    }

    docs = chunk_document(cleaned.text, metadata)

    # Deduplicate chunks
    chunk_texts = [d.page_content for d in docs]
    _, removed = deduplicate_chunks(chunk_texts)
    docs = [d for i, d in enumerate(docs) if i not in removed]

    # Add chunk index to metadata
    for i, doc in enumerate(docs):
        doc.metadata["chunk_index"] = i

    vectorstore = _get_vectorstore()
    vectorstore.add_documents(docs)

    return {
        "file_id": file_id,
        "file_name": original_filename,
        "total_chunks": len(docs) + len(removed),
        "stored_chunks": len(docs),
        "duplicates_removed": len(removed),
        "cleaning_stats": cleaned.stats,
    }


# ---------------------------------------------------------------------------
# Retrieval + Generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你是一个知识库助手。根据以下检索到的上下文回答问题。\n"
    "如果上下文中没有答案，就说不知道，不要编造。\n"
    "回答时在相关语句末尾标注引用来源，格式为 [来源: 文件名, 片段{{N}}]。\n\n"
    "上下文：\n{context}"
)


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        fn = d.metadata.get("file_name", "unknown")
        ci = d.metadata.get("chunk_index", "?")
        parts.append(f"[来源: {fn}, 片段#{ci}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def ask(question: str) -> dict:
    """Query the knowledge base and return answer + sources."""
    vectorstore = _get_vectorstore()

    try:
        count = len(vectorstore.get(limit=1).get("ids", []))
    except Exception:
        count = 0

    if count == 0:
        return {
            "answer": "知识库中还没有文档。请先上传文档。",
            "sources": [],
            "retrieved_chunks": [],
        }

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    # Retrieve once, use same docs for generation AND source display
    docs = retriever.invoke(question)
    context_text = _format_docs(docs)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "input": question})

    sources = []
    for i, doc in enumerate(docs):
        sources.append({
            "chunk_index": doc.metadata.get("chunk_index", i),
            "file_name": doc.metadata.get("file_name", "unknown"),
            "content_preview": doc.page_content[:200],
        })

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": sources,
    }


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------

def save_upload(file_content: bytes, filename: str) -> str:
    """Save uploaded file, return file_id."""
    file_id = uuid.uuid4().hex[:12]
    file_dir = DATA_DIR / file_id
    file_dir.mkdir(parents=True, exist_ok=True)
    file_path = file_dir / filename
    file_path.write_bytes(file_content)
    return file_id


def list_documents() -> List[dict]:
    """List all uploaded documents."""
    docs = []
    try:
        vs = _get_vectorstore()
    except Exception:
        vs = None

    for d in DATA_DIR.iterdir():
        if d.is_dir():
            files = list(d.iterdir())
            if files:
                indexed = False
                if vs is not None:
                    try:
                        result = vs.get(where={"file_id": d.name}, limit=1)
                        indexed = len(result.get("ids", [])) > 0
                    except Exception:
                        indexed = False
                docs.append({
                    "file_id": d.name,
                    "file_name": files[0].name,
                    "size_bytes": files[0].stat().st_size,
                    "indexed": indexed,
                })
    return docs


def delete_document(file_id: str) -> bool:
    """Delete a document and its vectors."""
    # Remove from Chroma
    try:
        vectorstore = _get_vectorstore()
        results = vectorstore.get(where={"file_id": file_id})
        if results and results.get("ids"):
            vectorstore.delete(ids=results["ids"])
    except Exception:
        pass

    # Remove from disk
    file_dir = DATA_DIR / file_id
    if file_dir.exists():
        shutil.rmtree(file_dir)
        return True
    return False
