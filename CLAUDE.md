# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal knowledge base RAG system — upload documents, ask questions, get AI-generated answers with source citations. A portfolio project for a first AI/LLM internship (Week 2-3 of an 8-week sprint). The goal is a project that interviewers can clone and run, or at minimum read the code and respect the engineering.

## Design Philosophy

- Fluid, iterative, not rigid or waterfall
- Easy over complex — no premature abstraction
- Built for brownfield (real-world messiness), not just greenfield
- Scalable from personal projects to enterprise

## Commands

```bash
# python -m venv <环境名称>
# <环境名称>\Scripts\activate
# 创建和激活虚拟环境
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Run Streamlit frontend
streamlit run streamlit_ui.py

# Run RAGAS evaluation
python evaluate.py
```

No linting configured yet.

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Backend | FastAPI | Async, type-safe, industry standard |
| RAG Framework | LangChain | Ecosystem maturity, interviewer familiarity |
| Vector DB | Chroma | Embedded, zero ops, persistent to `./chroma_db/` |
| LLM | DeepSeek V4 Flash via OpenRouter | Cost-effective, configured in `.env` |
| Embeddings | OpenAI text-embedding-3-small via OpenRouter | MTEB top performer (55.4% retrieval score) |
| PDF Parsing | `markitdown` or `unstructured` | Layout-aware, preserves tables/headings — NOT bare PyPDFLoader |
| Frontend | Streamlit | No HTML/CSS needed, interviewers can click around |
| Tracing | LangSmith | Project: `Personal_Knowledge_Database` |
| Eval | RAGAS | Faithfulness + Context Relevance + Answer Relevance |

## Architecture

```
═══════════════════════════════════════════════════════════════
                     OFFLINE (文档入库)
═══════════════════════════════════════════════════════════════

  用户上传 → FastAPI POST /upload
                │
                ▼
        ┌──────────────────┐
        │  DOCUMENT PARSER │  rag_engine.py
        │  markitdown /    │  PDF→Markdown, 保留表格/标题结构
        │  unstructured    │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  DATA CLEANING   │  rag_engine.py  ← 工业界必须有的一层
        │  · 文本规范化     │  30-40% 的原始文本是噪声
        │  · 页眉页脚过滤   │
        │  · 近重复检测     │  MinHash + LSH
        │  · 元数据提取     │  文件名、章节标题、页码
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │    CHUNKING      │  rag_engine.py
        │  RecursiveCharac- │  chunk_size=512 tokens, overlap=50-100
        │  terTextSplitter  │  保留父标题作为 chunk context prefix
        │  表格整表保留      │  表格 markdown 序列化,不切断
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  EMBEDDING +     │  rag_engine.py
        │  INDEXING        │  text-embedding-3-small via OpenRouter
        │  Chroma vector   │  持久化 ./chroma_db/
        │  + metadata      │  每个 chunk 存 source/chunk_index/section_title
        └──────────────────┘

═══════════════════════════════════════════════════════════════
                     ONLINE (用户提问)
═══════════════════════════════════════════════════════════════

  用户提问 → FastAPI POST /chat
                │
                ▼
        ┌──────────────────┐
        │ QUERY REWRITING  │  LLM 把模糊问题改写成精确检索 query
        │ (Week 3, 加分项)  │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ HYBRID RETRIEVAL │  向量 top-50 + BM25 top-50
        │ + RRF FUSION     │  Reciprocal Rank Fusion 融合
        │ (Week 3, 加分项)  │  → 精选到 top-20
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │    RERANKER      │  Cross-Encoder 重排序
        │ (Week 3, 加分项)  │  ms-marco-MiniLM-L6-v2
        │                  │  20 → 5, 最大单次精度提升(+30-40%)
        └────────┬─────────┘
                 │  top-5 chunks
                 ▼
        ┌──────────────────┐
        │  LLM GENERATION  │  DeepSeek V4 Flash via OpenRouter
        │  + CITATION      │  返回 {answer, sources[]}
        └────────┬─────────┘
                 │
                 ▼
        FastAPI Response → Streamlit UI 渲染

═══════════════════════════════════════════════════════════════

Streamlit UI ←→ FastAPI (HTTP)
```

## File Structure

```
personal_knowledge_rag/
├── app.py                # FastAPI entry point + API routes
├── rag_engine.py         # RAG core: parsing, cleaning, chunking, embedding, retrieval, generation
├── cleanup.py            # Data cleaning pipeline (normalization, dedup, metadata extraction)
├── streamlit_ui.py       # Streamlit frontend
├── evaluate.py           # RAGAS evaluation script
├── requirements.txt
├── .env.example          # Environment variable template (no secrets)
├── data/                 # Uploaded documents stored here
├── chroma_db/            # Chroma persistence directory
└── README.md
```

## API Routes

```
POST   /upload           # Upload file(s), returns file_id + parse status
POST   /chat             # Send question, returns {answer, sources, retrieved_chunks}
GET    /documents        # List uploaded documents with metadata
DELETE /documents/{id}   # Delete document + its vectors from Chroma
```

## LLM Configuration (Critical)

All LLM/Embedding calls go through OpenRouter. Every client must use `openai_api_base="https://openrouter.ai/api/v1"` and `OPENROUTER_API_KEY`. There is no `OPENAI_API_KEY`.

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "deepseek/deepseek-v4-flash"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)
```

## Implementation Plan: Two Tiers

### Tier 1 (Week 2 — must deliver)

The baseline that already beats 90% of intern projects:

1. **Document parser** — `markitdown` for PDF→Markdown, native loaders for MD/TXT
2. **Data cleaning** — `cleanup.py`: Unicode normalization, whitespace collapse, header/footer pattern filtering, basic dedup. This is the layer your instinct correctly identified as missing.
3. **Chunking** — `RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)` with section-title context prefix. Tables kept intact.
4. **Vector retrieval** — top_k=5 with metadata-rich Chroma collection
5. **LLM generation** — citation-grounded prompt, returns `{answer, sources}`
6. **RAGAS evaluation** — `evaluate.py` with 5-10 test queries, outputs Faithfulness + Context Relevance scores
7. **Streamlit UI** — upload tab + chat tab + source highlight

### Tier 2 (Week 3 — add if time permits)

Each item here is a concrete talking point in interviews:

1. **Hybrid retrieval** — maintain a BM25 index (`rank_bm25` library) alongside Chroma, fuse with RRF
2. **Cross-Encoder reranker** — `ms-marco-MiniLM-L6-v2` from HuggingFace, single largest accuracy gain
3. **Query rewriting** — LLM rewrites user question into retrieval-optimized query before search
4. **Reranker ablation** — `evaluate.py` compares with/without reranker, put the numbers in README

## Key Constraints

- `.env` is pre-configured. Never commit it. Create `.env.example` with placeholder values.
- LangSmith tracing is active — every chain execution logs to project `Personal_Knowledge_Database`
- This is a portfolio project: clean code, type hints, README with Mermaid architecture diagram + screenshots/GIF
- Target audience is internship interviewers — they may clone and run, or just read the code
- Don't over-engineer. Tier 1 complete > Tier 2 half-done.
