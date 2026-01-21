import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.config import BASE_DIR, LLM_MODEL, OLLAMA_BASE_URL, UPLOAD_DIR
from backend.describe_module import describe_image
from backend.ocr_module import get_ocr_service, initialize_ocr_service, run_ocr
from backend.ollama_client import OllamaClient
from backend.structure_module import generate_structure
from backend.utils import get_file_extension, save_upload_file


app = FastAPI(title="Image-to-Text Prototype") # creates the web server 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StructureRequest(BaseModel):
    filename: str
    mode: str  # table or mindmap


ollama_client: Optional[OllamaClient] = None


@app.on_event("startup")
async def on_startup():
    global ollama_client

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True) # ensures upload directory exists so that images can be saved there

    initialize_ocr_service(languages=["en"]) # load OCR reader once at startup

    ollama_client = OllamaClient(OLLAMA_BASE_URL) # initialize Ollama async client
    await ollama_client.startup()


@app.on_event("shutdown")
async def on_shutdown():
    if ollama_client is not None:
        await ollama_client.shutdown()


@app.get("/") # home page
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
    filename = f"ocr-{uuid.uuid4().hex}{ext}" # generates random file names 
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
        return {"description": description}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/structure")
async def structure_endpoint(file: UploadFile = File(...)):
    """
    Auto-Structures image data.
    Removed 'mode' parameter to match the new Auto-Frontend.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if ollama_client is None:
        raise HTTPException(status_code=500, detail="LLM client not initialized")

    ext = get_file_extension(file.filename or "")
    filename = f"struct-{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / filename

    try:
        save_upload_file(file, dest)
        # We no longer pass a 'mode', we just say "structure this"
        result = await generate_structure(dest, ollama_client)
        
        return {
            "mode": "auto",
            "result": result
        }
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()