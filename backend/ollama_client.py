from typing import Optional
import httpx


class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def startup(self):
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=120)

    async def shutdown(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def generate(
        self, model: str, prompt: str, images: Optional[list[str]] = None
    ) -> str:
        if self._client is None:
            await self.startup()

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if images:
            # Ollama multimodal API accepts base64 image strings in "images"
            payload["images"] = images

        try:
            resp = await self._client.post("/api/generate", json=payload)
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to reach Ollama at {self.base_url}: {e}") from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama error (status {resp.status_code}): {resp.text[:500]}"
            )

        data = resp.json()
        response_text = (data.get("response") or "").strip()
        if not response_text:
            raise RuntimeError("Empty response from Ollama")
        return response_text

