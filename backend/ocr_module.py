from pathlib import Path
from typing import Optional

import easyocr

class OCRService:
    def __init__(self, languages=None):
        if languages is None:
            languages = ["en"]
        self.reader = easyocr.Reader(languages, gpu=False) # forces to use CPU (slow but guarantees not to crash)

    def extract_text(self, image_path: Path) -> str:
        try:
            result = self.reader.readtext(str(image_path), detail=0) # detail=0 returns only text strings
            text = "\n".join(t.strip() for t in result if t and t.strip()) # stitches strings together into one big paragraph, separated by newlines.
            return text
        except Exception as e:
            raise RuntimeError(f"OCR failed: {e}") from e


# Global OCR service instance (saves time)
_ocr_service: Optional[OCRService] = None


def initialize_ocr_service(languages=None):
    """Initialize the global OCR service."""
    global _ocr_service
    if languages is None:
        languages = ["en"]
    _ocr_service = OCRService(languages=languages)
    return _ocr_service


def get_ocr_service() -> Optional[OCRService]:
    """Get the global OCR service instance."""
    return _ocr_service

# wrapper function
def run_ocr(image_path: Path) -> str:
    """Extract text from an image using OCR."""
    if _ocr_service is None:
        raise RuntimeError("OCR service not initialized")
    return _ocr_service.extract_text(image_path)
