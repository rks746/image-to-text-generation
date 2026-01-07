from pathlib import Path
import base64
import os
import shutil
from fastapi import UploadFile


def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)


def get_file_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1]
    return ext or ".png"


def encode_image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

