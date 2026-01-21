from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen2.5vl:3b")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
VLM_BASE_URL = "https://d8e006c5c8cc.ngrok-free.app/" #changes everytime the server restarts or Colab code is re-run
