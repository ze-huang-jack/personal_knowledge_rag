# Troubleshooting

Known errors and quick fixes for this project.

---

## 1. `ModuleNotFoundError: No module named 'langchain.chains'`

**Symptom:**

```
ModuleNotFoundError: No module named 'langchain.chains'
```

**Root cause:** LangChain >= 1.0 removed the `langchain.chains` module. `create_stuff_documents_chain` and `create_retrieval_chain` no longer exist. The correct approach is LCEL.

**Fix:**

```python
# ❌ Removed — will fail on langchain >= 1.0
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# ✅ LCEL — the standard way since langchain >= 1.0
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def _format_docs(docs):
    parts = []
    for d in docs:
        fn = d.metadata.get("file_name", "unknown")
        ci = d.metadata.get("chunk_index", "?")
        parts.append(f"[来源: {fn}, 片段#{ci}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

rag_chain = (
    {"context": retriever | _format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Retrieve + generate separately (so you can return sources)
docs = retriever.invoke(question)
answer = rag_chain.invoke(question)
```

**Files affected:** `rag_engine.py`, `evaluate.py`

**Pattern to remember:** Any LangChain tutorial using `create_*_chain` is pre-1.0. In 1.0+, always use `|` (LCEL) to compose chains. The pattern is always `retriever | _format_docs` for context, then `| prompt | llm | StrOutputParser()`.

---

## 2. `ModuleNotFoundError: No module named 'langchain_text_splitters'`

**Symptom:**

```
ModuleNotFoundError: No module named 'langchain_text_splitters'
```

**Fix:**

```bash
pip install langchain-text-splitters
```

The langchain ecosystem is split into many packages: `langchain`, `langchain-core`, `langchain-openai`, `langchain-chroma`, `langchain-text-splitters`, etc. Each needs its own `pip install`.

---

## 3. `markitdown` PDF conversion fails with `MissingDependencyException`

**Symptom:**

```
PdfConverter threw MissingDependencyException with message:
PdfConverter recognized the input as a potential .pdf file, but the dependencies
needed to read .pdf files have not been installed.
```

**Root cause:** `markitdown` was installed bare (`pip install markitdown`) without its PDF extras. PDF parsing needs `pdfminer-six` + `pdfplumber` which are optional dependencies.

**Fix:**

```bash
pip install "markitdown[pdf]"
```

Or install all optional deps:

```bash
pip install "markitdown[all]"
```

**Defense:** `rag_engine.py` has a three-layer PDF fallback:
1. `markitdown` (best quality, preserves tables/headings)
2. `pymupdf` / `fitz` (fast, already installed as `PyMuPDF`)
3. `pypdf` (pure Python, always available as `pypdf`)

If a layer fails, it logs the error and tries the next. This means the app won't crash just because one PDF library is misconfigured — it'll try the next available one.

---

## 4. Packages installed but venv still reports `ModuleNotFoundError`

**Symptom:** `pip install` succeeded but running the app still says module not found, or an older version of the module is used.

**Root cause:** You installed into system Python but the project runs in a virtual environment (`venv\`). They are isolated.

**Fix:** Always run pip from the venv:

```bash
# Windows
venv\Scripts\python.exe -m pip install <package>

# Or activate first
venv\Scripts\activate
pip install <package>
```

**Check which Python you're on:**
```bash
python -c "import sys; print(sys.executable)"
# Should print: ...\personal_knowledge_rag\venv\Scripts\python.exe
# NOT: C:\Users\suqiu\AppData\Local\Python\...
```

**Rule of thumb:** If you see `File "...\personal_knowledge_rag\venv\Lib\site-packages\..."` in a traceback, you're in the venv — install packages there.
