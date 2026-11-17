"""Detekce jazyka přes lingua-py (Apache-2.0, offline, bez služby).

Jazyky jsou omezené na `languages` (ISO 639-1); menší výběr = přesnější a rychlejší
rozhodování. Krátký text nebo text bez písmen jazyk nedostane.
"""

from __future__ import annotations

from importlib.metadata import version

from lingua import IsoCode639_1, Language, LanguageDetectorBuilder

from atr.ports.types import ComponentInfo
from atr.registry import register


@register("language", "lingua")
class LinguaLanguageDetector:
    def __init__(self, languages: list[str] | None = None, min_confidence: float = 0.5,
                 min_letters: int = 12) -> None:
        self.languages = languages or ["cs", "sk", "de", "en", "la", "pl", "fr"]
        self.min_confidence = min_confidence
        self.min_letters = min_letters
        langs = [Language.from_iso_code_639_1(getattr(IsoCode639_1, code.upper())) for code in self.languages]
        self._detector = LanguageDetectorBuilder.from_languages(*langs).with_preloaded_language_models().build()

    def detect(self, text: str) -> tuple[str, float] | None:
        if sum(ch.isalpha() for ch in text) < self.min_letters:
            return None
        values = self._detector.compute_language_confidence_values(text)
        if not values or values[0].value < self.min_confidence:
            return None
        return values[0].language.iso_code_639_1.name.lower(), round(values[0].value, 4)

    def info(self) -> ComponentInfo:
        return ComponentInfo("language", "lingua", version("lingua-language-detector"),
                             {"languages": self.languages, "min_confidence": self.min_confidence,
                              "min_letters": self.min_letters}, "Apache-2.0")
