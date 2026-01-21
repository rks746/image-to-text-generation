from pathlib import Path
from typing import TYPE_CHECKING
import httpx

from backend.config import VLM_BASE_URL, LLM_MODEL
from backend.ocr_module import get_ocr_service

if TYPE_CHECKING:
    from backend.ollama_client import OllamaClient

class StructureVLMClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False

    async def analyze_structure(self, image_path: Path) -> str:
        """
        Sends image to Colab VLM with task_type='auto'.
        The VLM decides if it's a table or diagram.
        """
        async with httpx.AsyncClient(timeout=180) as client:
            with open(image_path, "rb") as f:
                files = {"file": f}
                data = {"task_type": "auto"} 
                
                resp = await client.post(
                    f"{self.base_url}/analyze",
                    files=files,
                    data=data
                )

        if resp.status_code != 200:
            raise RuntimeError(f"VLM error: {resp.text}")

        return resp.json()["output"]

structure_vlm_client = StructureVLMClient(VLM_BASE_URL)

# FALLBACK PROMPT (Generic) 
def _get_auto_fallback_prompt(text: str) -> str:
    return f"""
    You are an expert data structurer.
    Analyze the text below. 
    - If it looks like a Table, output a JSON object with "columns" and "rows".
    - If it looks like a Process or Hierarchy, output a Markdown bullet list.
    - Output ONLY the structured data.
    
    Text: {text}
    """.strip()

# ORCHESTRATOR 
async def generate_structure(image_path: Path, ollama_client: "OllamaClient") -> str:
    """
    Primary: GPU-backed VLM (Auto Mode)
    Fallback: OCR + Generic LLM Prompt
    """
    
    # 1. Primary Path (Colab VLM)
    if await structure_vlm_client.health_check():
        print(f"Using VLM (Auto) path...")
        try:
            return await structure_vlm_client.analyze_structure(image_path)
        except Exception as e:
            print(f"VLM failed ({e}), switching to fallback...")

    # 2. Fallback Path (OCR + Local LLM)
    print("Using OCR fallback path...")
    ocr_service = get_ocr_service()
    if not ocr_service:
        raise RuntimeError("OCR service not initialized")

    extracted_text = ocr_service.extract_text(image_path)
    if not extracted_text.strip():
        return "No readable text found."

    # Use the generic prompt since we don't know the mode
    prompt = _get_auto_fallback_prompt(extracted_text)
    
    return await ollama_client.generate(model=LLM_MODEL, prompt=prompt)
