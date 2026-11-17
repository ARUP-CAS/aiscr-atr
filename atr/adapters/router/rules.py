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
    3. verdikt `blank` → `empty`;
    4. ostatní → `ocr` s verdiktem textové vrstvy jako důvodem.
    """

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
        if judgement.verdict is TextLayerVerdict.BLANK:
            # Vestavěný verdikt `blank` nevydává (bez textu nejde sken od prázdné strany
            # rozlišit, prázdnost pozná potrubí podle rozvržení); jiný adaptér verdiktu může.
            return RouteDecision(n, Route.EMPTY, judgement.reason)
        return RouteDecision(n, Route.OCR, f"text_layer_{judgement.verdict.value}: {judgement.reason}")

    def info(self) -> ComponentInfo:
        return ComponentInfo("router", "rules", __version__)
