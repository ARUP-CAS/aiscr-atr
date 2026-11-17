"""Verdikt textové vrstvy podle jednoduchých pravidel.

První verze. Prahy jsou v konfiguraci; kalibrace na korpusu AMČR patří do fáze 3.
"""

from __future__ import annotations

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


def _is_bad_char(ch: str) -> bool:
    if ch == "�":
        return True
    cat = unicodedata.category(ch)
    return cat in ("Co", "Cs", "Cn") or (cat == "Cc" and ch not in "\n\t")


@register("text_layer_judge", "rules")
class RulesTextLayerJudge:
    def __init__(
        self,
        min_chars: int = 20,
        max_bad_char_ratio: float = 0.05,
        min_alnum_ratio: float = 0.5,
        min_word_len_mean: float = 2.0,
    ) -> None:
        self.min_chars = min_chars
        self.max_bad_char_ratio = max_bad_char_ratio
        self.min_alnum_ratio = min_alnum_ratio
        self.min_word_len_mean = min_word_len_mean

    def judge(self, layer: TextLayer, geometry: PageGeometry) -> TextLayerJudgement:
        chars = [c for c in layer.text if not c.isspace()]
        if not chars:
            # Bez textové vrstvy nevíme, zda je strana prázdná, nebo je to sken.
            # O prázdnotě rozhodne až router (případně s klasifikací stran);
            # tady hlásíme jen to, co víme.
            return TextLayerJudgement(TextLayerVerdict.NONE, "no_text_layer", 0.0)

        bad = sum(_is_bad_char(c) for c in chars) / len(chars)
        if bad > self.max_bad_char_ratio:
            return TextLayerJudgement(TextLayerVerdict.GARBLED, f"bad_char_ratio={bad:.2f}", 1 - bad)

        alnum = sum(c.isalnum() for c in chars) / len(chars)
        words = layer.text.split()
        mean_len = sum(len(w) for w in words) / len(words) if words else 0.0
        if len(chars) < self.min_chars:
            return TextLayerJudgement(TextLayerVerdict.DIGITAL, f"short_text chars={len(chars)}", alnum)
        if alnum < self.min_alnum_ratio or mean_len < self.min_word_len_mean:
            return TextLayerJudgement(
                TextLayerVerdict.GARBLED, f"alnum_ratio={alnum:.2f} mean_word_len={mean_len:.1f}", alnum
            )
        return TextLayerJudgement(TextLayerVerdict.DIGITAL, "text_layer_ok", alnum)

    def info(self) -> ComponentInfo:
        return ComponentInfo(
            "text_layer_judge",
            "rules",
            __version__,
            {
                "min_chars": self.min_chars,
                "max_bad_char_ratio": self.max_bad_char_ratio,
                "min_alnum_ratio": self.min_alnum_ratio,
                "min_word_len_mean": self.min_word_len_mean,
            },
        )
