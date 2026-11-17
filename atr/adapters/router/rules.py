"""Směrování stran podle pravidel."""

from __future__ import annotations

from atr import __version__
from atr.ports.types import (
    ComponentInfo,
    PageClass,
    PageGeometry,
    Route,
    RouteDecision,
    TextLayerJudgement,
    TextLayerVerdict,
)
from atr.registry import register


@register("router", "rules")
class RulesRouter:
    """Pořadí pravidel:

    1. vynucená strana → `forced` s důvodem od volajícího;
    2. použitelná textová vrstva → `digital`;
    3. prázdná strana → `empty` (jen když prázdnotu nehlásí ani klasifikace stran);
    4. ostatní → `ocr` s verdiktem textové vrstvy jako důvodem.
    """

    def __init__(self, blank_classes: list[str] | None = None) -> None:
        # Třídy klasifikace stran, které znamenají prázdnou stranu. Bez klasifikace
        # se o prázdnotě rozhoduje jen podle textové vrstvy.
        self.blank_classes = set(blank_classes or [])

    def decide(
        self,
        geometry: PageGeometry,
        judgement: TextLayerJudgement,
        forced_reason: str | None,
        page_class: PageClass | None,
    ) -> RouteDecision:
        n = geometry.number
        if forced_reason is not None:
            return RouteDecision(n, Route.FORCED, forced_reason)
        if judgement.verdict is TextLayerVerdict.DIGITAL:
            return RouteDecision(n, Route.DIGITAL, judgement.reason)
        if judgement.verdict is TextLayerVerdict.BLANK and (
            page_class is None or not self.blank_classes or page_class.label in self.blank_classes
        ):
            return RouteDecision(n, Route.EMPTY, judgement.reason)
        return RouteDecision(n, Route.OCR, f"text_layer_{judgement.verdict.value}: {judgement.reason}")

    def info(self) -> ComponentInfo:
        return ComponentInfo("router", "rules", __version__, {"blank_classes": sorted(self.blank_classes)})
