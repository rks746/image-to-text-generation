from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.ollama_client import OllamaClient

from backend.ocr_module import get_ocr_service


async def describe_image(image_path: Path, ollama_client: "OllamaClient") -> str:
    """
    Generate a natural language description of an image.
    Uses OCR to extract text, then text-only LLM to generate description.
    """
    ocr_service = get_ocr_service()
    if ocr_service is None:
        raise RuntimeError("OCR service not initialized")

    # 1. OCR
    extracted_text = ocr_service.extract_text(image_path)

    if not extracted_text.strip():
        return "No readable text found in the image."

    # 2. Text-only LLM
    prompt = f"""
You are an assistant that explains notes clearly.

Based on the following extracted text, generate a clear natural-language
description explaining the topic and key ideas.

Extracted text:
{extracted_text}
""".strip()

    response = await ollama_client.generate(
        model="qwen2.5:1.5b",
        prompt=prompt
    )

    return response

