"""Rozpoznávač bloků za rozhraním kompatibilním s OpenAI (vLLM, SGLang)."""

from __future__ import annotations

import base64

import httpx

from atr.ports.errors import ProcessingError, classify_http_error
from atr.ports.types import ComponentInfo, LayoutBlock, PageImage, RecognizedContent
from atr.registry import register


@register("recognizer", "openai")
class OpenAIBlockRecognizer:
    """Pošle výřez bloku s pokynem a vrátí text.

    `prompt` je pokyn modelu pro daný typ úlohy, např. "OCR:" nebo
    "Table Recognition:" (PaddleOCR-VL), "Text Recognition:" (GLM-OCR).
    Jeden rozpoznávač = jedna úloha; tabulka typ bloku → rozpoznávač je
    v konfiguraci potrubí.
    """

    def __init__(
        self,
        url: str,
        model: str,
        prompt: str,
        output_format: str = "text",
        max_tokens: int = 4096,
        timeout_s: float = 120.0,
        licence: str | None = None,
        api_key: str = "EMPTY",
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.prompt = prompt
        self.output_format = output_format
        self.max_tokens = max_tokens
        self.licence = licence
        self.client = httpx.Client(timeout=timeout_s, headers={"Authorization": f"Bearer {api_key}"})

    def recognize(self, crop: PageImage, block: LayoutBlock) -> RecognizedContent:
        image_url = "data:image/png;base64," + base64.b64encode(crop.png).decode("ascii")
        try:
            resp = self._post(image_url)
        except httpx.HTTPError as exc:
            raise classify_http_error("recognizer", f"{self.url} ({self.model})", exc) from exc
        choice = resp.json()["choices"][0]
        finish = choice.get("finish_reason")
        if finish not in (None, "stop"):
            # Výstup useknutý limitem tokenů (nebo jinak nedokončený) není úplný výsledek.
            raise ProcessingError("recognition_incomplete",
                                  f"model {self.model} skončil s finish_reason={finish!r}")
        text = choice["message"]["content"] or ""
        return RecognizedContent(text=text.strip(), format=self.output_format)

    def _post(self, image_url: str) -> httpx.Response:
        resp = self.client.post(
            f"{self.url}/chat/completions",
            json={
                "model": self.model,
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": self.prompt},
                        ],
                    }
                ],
            },
        )
        resp.raise_for_status()
        return resp

    def info(self) -> ComponentInfo:
        return ComponentInfo(
            "recognizer",
            self.model,
            "served",
            {"url": self.url, "prompt": self.prompt, "output_format": self.output_format},
            self.licence,
        )
