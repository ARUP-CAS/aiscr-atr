"""Skladatel výstupu `atr/docproc-json`.

PROZATÍMNÍ TVAR. Normativní schéma vzniká v aiscr-docs-pipeline#17. Tady je
jen to, co z kontraktu už víme (routing každé strany, bloky s typem, polohou,
pořadím a textem, nepovinné `lines[]` s přesností polohy, verdikt textové
vrstvy, žádná data o běhu). Až schéma bude, tvar se doladí a výstup se bude
validovat (komponenta 14).
"""

from __future__ import annotations

from atr import __version__
from atr.ports.types import ComponentInfo, DocumentResult, Route
from atr.registry import register

CONTRACT_VERSION = "0.0.0-draft"


def _bbox(b) -> list[float]:
    return [round(b.x0, 2), round(b.y0, 2), round(b.x1, 2), round(b.y1, 2)]


@register("assembler", "docproc")
class DocprocAssembler:
    def assemble(self, result: DocumentResult) -> dict:
        routing, pages = [], []
        for page in result.pages:
            g, d = page.geometry, page.decision
            entry = {"page": g.number, "route": d.route.value, "reason": d.reason}
            if d.route is Route.FORCED:
                entry["forced_reason"] = d.reason
            if page.judgement is not None:
                entry["text_layer"] = page.judgement.verdict.value
            if page.page_class is not None:
                entry["label"] = page.page_class.label
                entry["label_confidence"] = round(page.page_class.confidence, 4)
            routing.append(entry)

            if d.route is Route.EMPTY:
                continue  # prázdná strana nemá záznam v pages[], jen v routing
            out = {
                "pageNumber": g.number,
                "width": round(g.width, 2),
                "height": round(g.height, 2),
                "rotation": g.rotation,
                "coordinateUnit": "pt",
                "blocks": [],
            }
            if g.source_pixels is not None:
                out["sourcePixels"] = list(g.source_pixels)
            if page.language:
                out["language"] = page.language
            for block in sorted(page.blocks, key=lambda b: b.order):
                b = {
                    "type": block.type.value,
                    "bbox": _bbox(block.bbox),
                    "confidence": round(block.confidence, 4),
                    "readingOrder": block.order,
                    "content": block.text,
                    "contentFormat": block.format,
                }
                if block.language:
                    b["language"] = block.language
                if block.lines:
                    b["lines"] = [
                        {
                            "text": l.text,
                            "bbox": _bbox(l.bbox) if l.bbox else None,
                            "position": l.accuracy.value,
                        }
                        for l in block.lines
                    ]
                out["blocks"].append(b)
            pages.append(out)

        out = {
            "version": CONTRACT_VERSION,
            "pageCount": len(result.pages),
            "metadata": {"routing": {"pages": routing}},
            "pages": pages,
        }
        language = _document_language(result)
        if language:
            out["documentLanguage"] = language
        return out

    def info(self) -> ComponentInfo:
        return ComponentInfo("assembler", "docproc", __version__, {"contract_version": CONTRACT_VERSION})


def _document_language(result: DocumentResult) -> str | None:
    """Jazyk s největším množstvím textu napříč bloky."""
    weight: dict[str, int] = {}
    for page in result.pages:
        for block in page.blocks:
            if block.language:
                weight[block.language] = weight.get(block.language, 0) + len(block.text)
    return max(weight, key=weight.get) if weight else None
