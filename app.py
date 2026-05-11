from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

from rag_engine import save_upload, index_file, ask, list_documents, delete_document

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB

app = FastAPI(title="Personal Knowledge Base RAG", version="0.1.0",)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

class ChatRequest(BaseModel):
    question: str



@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    """Upload a document, parse and index it into the knowledge base."""
    print(f"[UPLOAD] Received file: {file.filename}, content-length: {request.headers.get('content-length', 'N/A')}")
    if not file.filename:
        raise HTTPException(400, "No file provided")

    # Reject oversized uploads before reading into memory
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Maximum {MAX_UPLOAD_BYTES // 1024 // 1024}MB")

    allowed = (".pdf", ".md", ".markdown", ".txt", ".text")
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(allowed)}")

    contents = await file.read()
    print(f"[UPLOAD] Read {len(contents)} bytes")
    if len(contents) == 0:
        raise HTTPException(400, "Empty file")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Maximum {MAX_UPLOAD_BYTES // 1024 // 1024}MB")

    file_id = save_upload(contents, file.filename)
    print(f"[UPLOAD] Saved as file_id={file_id}, starting index...")
    try:
        result = index_file(file_id)
        print(f"[UPLOAD] Index complete: {result['stored_chunks']} chunks stored")
        return result
    except Exception as e:
        print(f"[UPLOAD] Index failed, rolling back: {e}")
        delete_document(file_id)  # rollback: remove saved file
        raise


@app.post("/chat")
async def chat(req: ChatRequest):
    """Ask a question against the knowledge base."""
    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    return ask(req.question)


@app.get("/documents")
async def get_documents():
    """List all uploaded documents."""
    return {"documents": list_documents()}


@app.delete("/documents/{file_id}")
async def remove_document(file_id: str):
    """Delete a document and its vectors."""
    success = delete_document(file_id)
    if not success:
        raise HTTPException(404, "Document not found")
    return {"deleted": file_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
