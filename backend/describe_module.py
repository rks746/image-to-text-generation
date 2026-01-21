from pathlib import Path
from typing import TYPE_CHECKING
import httpx

from backend.config import VLM_BASE_URL, LLM_MODEL
from backend.ocr_module import get_ocr_service

if TYPE_CHECKING:
    from backend.ollama_client import OllamaClient


class VLMClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False

    async def describe_image(self, image_path: Path) -> str:
        async with httpx.AsyncClient(timeout=180) as client:
            with open(image_path, "rb") as f:
                files = {"file": f}
                data = {"task_type": "text"} # text description
                
                resp = await client.post(
                    f"{self.base_url}/analyze",
                    files=files,
                    data=data
                ) # transmits the image data through the ngrok tunnel to Google's servers

        if resp.status_code != 200:
            raise RuntimeError(f"VLM error: {resp.text}")

        return resp.json()["output"]

vlm_client = VLMClient(VLM_BASE_URL)

# calls VLMCLient
# activates backup plan if fails 
async def describe_image(image_path: Path, ollama_client: "OllamaClient") -> str:
    # 1. Primary Path
    if await vlm_client.health_check():
        try:
            return await vlm_client.describe_image(image_path)
        except Exception as e:
            print("VLM failed, falling back:", e)

    # 2. Fallback Path (OCR)
    print("Using OCR fallback...")
    ocr = get_ocr_service()
    # SAFETY CHECK: Ensure OCR is actually running
    if ocr is None:
        raise RuntimeError("OCR service not initialized. Cannot perform fallback.")
    text = ocr.extract_text(image_path)
    if not text.strip(): return "No text found."
    
    prompt = f"Explain this text clearly:\n{text}"
    return await ollama_client.generate(model=LLM_MODEL, prompt=prompt)
