"""Testy jednotlivých adaptérů bez potrubí."""

from atr.adapters.router.rules import RulesRouter
from atr.adapters.text_assigner.overlap import OverlapTextAssigner
from atr.adapters.text_layer_judge.rules import RulesTextLayerJudge
from atr.ports.types import (
    BBox,
    BlockType,
    LayoutBlock,
    Line,
    PageGeometry,
    Route,
    TextLayer,
    TextLayerJudgement,
    TextLayerVerdict,
    Word,
)

G = PageGeometry(1, 595, 842, 0)


def layer(*lines: tuple[str, tuple[float, float, float, float]]) -> TextLayer:
    out = []
    for text, (x0, y0, x1, y1) in lines:
        tokens = text.split()
        step = (x1 - x0) / len(tokens)
        words = tuple(Word(t, BBox(x0 + i * step, y0, x0 + (i + 1) * step, y1)) for i, t in enumerate(tokens))
        out.append(Line(text, BBox(x0, y0, x1, y1), words))
    return TextLayer(1, tuple(out))


def test_judge_verdicts():
    judge = RulesTextLayerJudge()
    assert judge.judge(TextLayer(1, ()), G).verdict is TextLayerVerdict.NONE
    good = layer(("Nálezová zpráva o výzkumu sídliště z doby bronzové", (0, 0, 400, 12)))
    assert judge.judge(good, G).verdict is TextLayerVerdict.DIGITAL
    bad = layer(("   abc", (0, 0, 400, 12)))
    assert judge.judge(bad, G).verdict is TextLayerVerdict.GARBLED


def test_router_priority():
    router = RulesRouter()
    ok = TextLayerJudgement(TextLayerVerdict.DIGITAL, "ok")
    assert router.decide(G, ok, "needs_atr", None).route is Route.FORCED
    assert router.decide(G, ok, None, None).route is Route.DIGITAL
    none = TextLayerJudgement(TextLayerVerdict.NONE, "no_text_layer")
    assert router.decide(G, none, None, None).route is Route.OCR


def test_assigner_splits_words_into_blocks_and_keeps_orphans():
    text = layer(("Levý sloupec", (50, 100, 250, 112)), ("Pravý sloupec", (320, 100, 520, 112)),
                 ("Poznámka mimo", (50, 800, 200, 812)))
    blocks = [
        LayoutBlock(BlockType.TEXT, BBox(45, 95, 260, 120), 0.9, 0),
        LayoutBlock(BlockType.TEXT, BBox(315, 95, 525, 120), 0.9, 1),
        LayoutBlock(BlockType.FIGURE, BBox(40, 400, 500, 700), 0.9, 2),
    ]
    out = OverlapTextAssigner().assign(text, blocks)
    assert [b.text for b in out] == ["Levý sloupec", "Pravý sloupec", "", "Poznámka mimo"]
    assert out[3].type is BlockType.TEXT  # text mimo bloky se neztratí
