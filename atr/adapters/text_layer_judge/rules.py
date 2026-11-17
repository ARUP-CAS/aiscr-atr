"""Verdikt textové vrstvy: použitelná (digital), vadná (garbled), starší OCR (ocr), žádná (none).

Kontroly jsou obecné, nezávislé na typu dokumentu, a jdou v tomto pořadí:

1. žádný text → `none`;
2. **starší OCR vrstva**: většina znaků je neviditelná (PDF režim vykreslení 3) a strana
   je převážně obrázek. Takové vrstvě nevěříme a strana jde znovu na OCR → `ocr`;
3. **vadné znaky**: soukromá oblast Unicode, náhradní znak, řídicí znaky, znaky bez
   mapování na Unicode a značky `(cid:N)` → `garbled`;
4. **chybné kódování** (mojibake, např. „Å¾“ místo „ž“) → `garbled`;
5. **rozpadlá slova**: text rozsekaný na jednotlivá písmena („N á l e z“) → `garbled`;
6. **slepená slova**: chybí mezery, slova jsou nepřirozeně dlouhá → `garbled`;
7. **nesmyslná slova**: velký podíl „slov“ bez samohlásky (r a l se počítají, kvůli
   „vlk“, „prst“) nebo s nesmyslnými znaky uprostřed → `garbled`;
8. **příliš mnoho symbolů** proti písmenům a číslicím → `garbled`;
9. jinak `digital`.

Krátký text (méně než `min_chars` znaků) se hodnotí jen kontrolami 1–3, ostatní by
na pár slovech dávaly náhodné výsledky. Prahy jsou v konfiguraci.
"""

from __future__ import annotations

import re
import unicodedata

from atr import __version__
from atr.ports.types import (
    ComponentInfo,
    PageGeometry,
    TextLayer,
    TextLayerJudgement,
    TextLayerVerdict,
)
from atr.registry import register

_CID = re.compile(r"\(cid:\d+\)")
# Typické dvojice znaků z UTF-8 přečteného jako Latin-1/CP1250 (ž → Å¾, á → Ã¡, ě → Ä›, ’ → â€™).
_MOJIBAKE = re.compile(r"[ÃÅÄÂâ][\u0080-¿Ā-ſ‘-›€™]")
_VOWELS = set("aeiouyáéěíóúůýäëïöüœæàèìòùâêîôûåørlŕĺ")
_TOKEN = re.compile(r"\S+")
_WORD_CHARS = re.compile(r"^[^\W\d_]+$")  # jen písmena
# Opakované znaky (oddělovací čáry „____“, tečkované obsahy „.....“) a odkazy se nehodnotí.
_RUNS = re.compile(r"(\S)\1{3,}")
_LINKS = re.compile(r"\S*(?:https?://|www\.|@)\S*|\S+\.(?:cz|com|org|eu|net)\S*", re.IGNORECASE)


def _bad_char(ch: str) -> bool:
    if ch == "�":
        return True
    cat = unicodedata.category(ch)
    return cat in ("Co", "Cs", "Cn") or (cat == "Cc" and ch not in "\n\t")


@register("text_layer_judge", "rules")
class RulesTextLayerJudge:
    def __init__(
        self,
        min_chars: int = 80,
        max_bad_char_ratio: float = 0.05,
        ocr_invisible_ratio: float = 0.8,
        ocr_image_coverage: float = 0.5,
        max_mojibake_ratio: float = 0.02,
        max_single_letter_ratio: float = 0.35,
        max_long_word_ratio: float = 0.08,
        long_word_len: int = 25,
        max_implausible_ratio: float = 0.25,
        max_symbol_ratio: float = 0.2,
        # Starší prahy zůstávají kvůli zpětné kompatibilitě konfigurace, nepoužívají se.
        min_alnum_ratio: float | None = None,
        min_word_len_mean: float | None = None,
    ) -> None:
        self.params = {
            "min_chars": min_chars,
            "max_bad_char_ratio": max_bad_char_ratio,
            "ocr_invisible_ratio": ocr_invisible_ratio,
            "ocr_image_coverage": ocr_image_coverage,
            "max_mojibake_ratio": max_mojibake_ratio,
            "max_single_letter_ratio": max_single_letter_ratio,
            "max_long_word_ratio": max_long_word_ratio,
            "long_word_len": long_word_len,
            "max_implausible_ratio": max_implausible_ratio,
            "max_symbol_ratio": max_symbol_ratio,
        }

    def judge(self, layer: TextLayer, geometry: PageGeometry) -> TextLayerJudgement:
        p = self.params
        text = layer.text
        chars = [c for c in text if not c.isspace()]
        n = len(chars)
        if n == 0:
            # Bez textu nejde rozlišit sken od prázdné strany; rozhodne router a rozvržení.
            return TextLayerJudgement(TextLayerVerdict.NONE, "no_text_layer", 0.0)

        invisible = layer.invisible_chars / n
        if invisible >= p["ocr_invisible_ratio"] and layer.image_coverage >= p["ocr_image_coverage"]:
            return TextLayerJudgement(
                TextLayerVerdict.OCR,
                f"invisible_text={invisible:.2f} image_coverage={layer.image_coverage:.2f}", 0.0)

        bad = (sum(_bad_char(c) for c in chars) + layer.unmapped_chars
               + len(_CID.findall(text)) * 5) / n
        if bad > p["max_bad_char_ratio"]:
            return TextLayerJudgement(TextLayerVerdict.GARBLED, f"bad_char_ratio={bad:.3f}", 1 - bad)

        if n < p["min_chars"]:
            return TextLayerJudgement(TextLayerVerdict.DIGITAL, f"short_text chars={n}", None)

        clean = _LINKS.sub(" ", _RUNS.sub(" ", text))
        tokens = _TOKEN.findall(clean.replace("/", " "))
        words = [t.strip(".,;:!?()[]{}\"'„“”‚‘’«»-–—/") for t in tokens]
        words = [w for w in words if w]
        letter_words = [w for w in words if _WORD_CHARS.match(w)]

        mojibake = len(_MOJIBAKE.findall(clean)) / max(len(words), 1)
        if mojibake > p["max_mojibake_ratio"]:
            return TextLayerJudgement(TextLayerVerdict.GARBLED, f"mojibake_ratio={mojibake:.3f}", 1 - mojibake)

        if letter_words:
            single = sum(len(w) == 1 for w in letter_words) / len(letter_words)
            # Jednopísmenné předložky a spojky (v, s, k, z, a, i, o, u) jsou běžné; rozpadlý text
            # má jednopísmenných „slov“ většinu.
            if single > p["max_single_letter_ratio"]:
                return TextLayerJudgement(TextLayerVerdict.GARBLED, f"single_letter_ratio={single:.2f}", 1 - single)

        long_words = sum(len(w) >= p["long_word_len"] for w in words) / max(len(words), 1)
        if long_words > p["max_long_word_ratio"]:
            return TextLayerJudgement(TextLayerVerdict.GARBLED, f"long_word_ratio={long_words:.2f}", 1 - long_words)

        candidates = [w.lower() for w in words if len(w) >= 3 and any(ch.isalpha() for ch in w)
                      and not any(ch.isdigit() for ch in w)]
        if candidates:
            implausible = sum(_implausible(w) for w in candidates) / len(candidates)
            if implausible > p["max_implausible_ratio"]:
                return TextLayerJudgement(
                    TextLayerVerdict.GARBLED, f"implausible_word_ratio={implausible:.2f}", 1 - implausible)

        kept = [c for c in clean if not c.isspace()]
        symbols = sum(not (c.isalnum() or c in ".,;:!?()-–—−\"'„“”/%§+=*&°•●·") for c in kept) / max(len(kept), 1)
        if symbols > p["max_symbol_ratio"]:
            return TextLayerJudgement(TextLayerVerdict.GARBLED, f"symbol_ratio={symbols:.2f}", 1 - symbols)

        return TextLayerJudgement(TextLayerVerdict.DIGITAL, "text_layer_ok", 1.0)

    def info(self) -> ComponentInfo:
        return ComponentInfo("text_layer_judge", "rules", __version__, dict(self.params))


def _implausible(word: str) -> bool:
    """Slovo bez samohlásky (včetně slabičného r/l), nebo se symbolem uprostřed."""
    letters = [ch for ch in word if ch.isalpha()]
    if not letters:
        return True
    if not any(ch in _VOWELS for ch in letters):
        return True
    inner = word[1:-1]
    return any(not (ch.isalpha() or ch in "-'’.") for ch in inner)
