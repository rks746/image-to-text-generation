from pathlib import Path
from typing import TYPE_CHECKING
import httpx

from backend.config import VLM_BASE_URL
from backend.config import LLM_MODEL
from backend.ocr_module import get_ocr_service

if TYPE_CHECKING:
    from backend.ollama_client import OllamaClient


class VLMClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def health_check(self) -> bool:
        print("Checking VLM health at:", f"{self.base_url}/health")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.base_url}/health")
                print("Health status:", resp.status_code)
                print("Health response:", resp.text)
                return resp.status_code == 200
        except Exception as e:
            print("Health check failed:", e)
            return False

    async def describe_image(self, image_path: Path) -> str:
        async with httpx.AsyncClient(timeout=180) as client:
            with open(image_path, "rb") as f:
                files = {"file": f}
                resp = await client.post(
                    f"{self.base_url}/describe-image",
                    files=files
                )

        if resp.status_code != 200:
            raise RuntimeError(f"VLM error: {resp.text}")

        return resp.json()["description"]


vlm_client = VLMClient(VLM_BASE_URL)


async def describe_image(image_path: Path, ollama_client: "OllamaClient") -> str:
    """
    Primary: GPU-backed VLM microservice
    Fallback: OCR + local text LLM
    """

    # --- Primary path: VLM ---
    if await vlm_client.health_check():
        print("Using VLM path...")
        try:
            return await vlm_client.describe_image(image_path)
        except Exception as e:
            print("VLM failed, falling back:", e)

    # --- Fallback path: OCR + LLM ---
    print("Using OCR fallback path...")

    ocr_service = get_ocr_service()
    if ocr_service is None:
        raise RuntimeError("OCR service not initialized")

    extracted_text = ocr_service.extract_text(image_path)

    if not extracted_text.strip():
        return "No readable text found in the image."

    prompt = f"""
You are an assistant that explains notes clearly.

Based on the following extracted text, generate a clear natural-language
description explaining the topic and key ideas.

Extracted text:
{extracted_text}
""".strip()

    response = await ollama_client.generate(
        model=LLM_MODEL,
        prompt=prompt
    )

    return response
