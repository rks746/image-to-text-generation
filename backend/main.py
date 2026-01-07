import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.config import BASE_DIR, LLM_MODEL, OLLAMA_BASE_URL, UPLOAD_DIR
from backend.describe_module import describe_image
from backend.ocr_module import get_ocr_service, initialize_ocr_service, run_ocr
from backend.ollama_client import OllamaClient
from backend.structure_module import build_mindmap, build_table
from backend.utils import get_file_extension, save_upload_file


app = FastAPI(title="Image Intelligence Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StructureRequest(BaseModel):
    text: str
    mode: str  # "table" or "mindmap"


ollama_client: Optional[OllamaClient] = None


@app.on_event("startup")
async def on_startup():
    global ollama_client

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Load OCR reader once at startup
    initialize_ocr_service(languages=["en"])

    # Initialize Ollama async client
    ollama_client = OllamaClient(OLLAMA_BASE_URL)
    await ollama_client.startup()


@app.on_event("shutdown")
async def on_shutdown():
    if ollama_client is not None:
        await ollama_client.shutdown()


@app.get("/")
async def serve_index():
    index_path = BASE_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="index.html not found on server")
    return FileResponse(index_path)


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Save uploaded image to /uploads and return the filename.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    ext = get_file_extension(file.filename or "")
    filename = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / filename

    try:
        save_upload_file(file, dest)
    finally:
        file.file.close()

    return {"filename": filename, "path": f"/uploads/{filename}"}


@app.post("/extract-text")
async def extract_text_endpoint(file: UploadFile = File(...)):
    """
    Extract text from an uploaded image using EasyOCR.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    ocr_service = get_ocr_service()
    if ocr_service is None:
        raise HTTPException(status_code=500, detail="OCR service not initialized")

    ext = get_file_extension(file.filename or "")
    filename = f"ocr-{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / filename

    try:
        save_upload_file(file, dest)
    finally:
        file.file.close()

    try:
        text = run_ocr(dest)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"text": text}


@app.post("/describe-image")
async def describe_image_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if ollama_client is None:
        raise HTTPException(status_code=500, detail="LLM client not initialized")

    ext = get_file_extension(file.filename or "")
    filename = f"desc-{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / filename

    try:
        save_upload_file(file, dest)
    finally:
        file.file.close()

    try:
        description = await describe_image(dest, ollama_client)
        if not description.strip():
            return {"description": "No readable text found in the image."}
        return {"description": description}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/structure")
async def structure_endpoint(req: StructureRequest):
    """
    Convert extracted text into structured formats using qwen2.5:1.5b.

    - mode = \"table\"   -> JSON table
    - mode = \"mindmap\" -> Markdown bullet hierarchy
    """
    if ollama_client is None:
        raise HTTPException(status_code=500, detail="LLM client not initialized")

    mode = req.mode.lower().strip()
    if mode not in {"table", "mindmap"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Must be 'table' or 'mindmap'.",
        )

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    try:
        if mode == "table":
            structured = await build_table(req.text, ollama_client, LLM_MODEL)
        else:
            structured = await build_mindmap(req.text, ollama_client, LLM_MODEL)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # For table we expect JSON; for mindmap we expect Markdown text.
    return {"mode": mode, "result": structured}

