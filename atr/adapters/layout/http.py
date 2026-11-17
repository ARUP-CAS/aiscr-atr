"""Klient kontejneru s modelem rozvržení (protokol v docs/model-services.md)."""

from __future__ import annotations

import base64

import httpx

from atr.ports.errors import DependencyError
from atr.ports.types import (
    BBox,
    BlockType,
    ComponentInfo,
    LayoutBlock,
    PageGeometry,
    PageImage,
)
from atr.registry import register


@register("layout", "http")
class HttpLayoutDetector:
    """Posílá obrázek strany na `POST {url}/predict`, dostává bloky v pixelech
    obrázku a převádí je na body zobrazené strany. Třídy modelu mapuje na
    `BlockType` podle `label_map` (výchozí mapa je u služby v `/info`)."""

    def __init__(
        self,
        url: str,
        label_map: dict[str, str] | None = None,
        min_confidence: float = 0.3,
        timeout_s: float = 60.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.label_map = {k: BlockType(v) for k, v in (label_map or {}).items()}
        self.min_confidence = min_confidence
        self.client = httpx.Client(timeout=timeout_s)
        self._info: dict | None = None

    def detect(self, image: PageImage, geometry: PageGeometry) -> list[LayoutBlock]:
        try:
            resp = self.client.post(
                f"{self.url}/predict",
                json={"image_png_base64": base64.b64encode(image.png).decode("ascii")},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise DependencyError("layout", f"{self.url}: {exc}") from exc
        k = image.px_to_pt
        blocks = []
        for item in resp.json()["blocks"]:
            if item["confidence"] < self.min_confidence:
                continue
            x0, y0, x1, y1 = item["bbox_px"]
            label = item["label"]
            blocks.append(
                LayoutBlock(
                    type=self.label_map.get(label, BlockType.OTHER),
                    bbox=BBox(x0 * k, y0 * k, x1 * k, y1 * k),
                    confidence=float(item["confidence"]),
                    order=item.get("order"),
                    source_label=label,
                )
            )
        return blocks

    def info(self) -> ComponentInfo:
        if self._info is None:
            try:
                resp = self.client.get(f"{self.url}/info", timeout=5.0)
                resp.raise_for_status()
                self._info = resp.json()
            except (httpx.HTTPError, ValueError):
                # Nedostupná služba nesmí shodit /info; zkusí se znovu příště.
                return ComponentInfo("layout", "unreachable", "unknown", {"url": self.url})
        return ComponentInfo(
            "layout",
            self._info.get("model", "unknown"),
            self._info.get("version", "unknown"),
            {"url": self.url, "min_confidence": self.min_confidence},
            self._info.get("licence"),
        )
