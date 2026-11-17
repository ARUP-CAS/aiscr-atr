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


def _lb(t: str, box, order=None) -> LayoutBlock:
    return LayoutBlock(BlockType(t), BBox(*box), 0.9, order)


def test_refiner_drops_nested_duplicates_and_splits_overlaps():
    from atr.adapters.layout_refiner.overlaps import OverlapRefiner

    blocks = [
        _lb("text", (100, 100, 300, 110)),   # řádek 1
        _lb("text", (100, 109, 300, 119)),   # řádek 2, překrývá řádek 1 o 1 pt → dělit
        _lb("text", (100, 140, 300, 150)),   # řádky 3 a 4 přes sebe z poloviny → sloučit
        _lb("text", (100, 145, 300, 155)),
        _lb("table", (50, 200, 500, 400)),
        _lb("text", (60, 210, 200, 230)),    # leží celý v tabulce → pryč
        _lb("figure", (50, 500, 300, 700)),
        _lb("text", (60, 510, 200, 530)),    # v obrázku zůstává (obrázek se nečte)
    ]
    out = OverlapRefiner().refine(blocks, G)
    assert [b.type.value for b in out] == ["text", "text", "text", "table", "figure", "text"]
    assert out[0].bbox.y1 == out[1].bbox.y0 == 109.5  # malý překryv rozdělen napůl
    assert out[0].bbox.intersection(out[1].bbox) == 0
    assert (out[2].bbox.y0, out[2].bbox.y1) == (140, 155)  # výrazný překryv sloučen


def test_xycut_reads_columns_and_sections():
    from atr.adapters.reading_order.adapters import XYCutReadingOrder

    blocks = [
        _lb("page_footer", (250, 800, 350, 810)),
        _lb("text", (320, 150, 520, 400)),   # pravý sloupec
        _lb("title", (100, 80, 500, 110)),
        _lb("text", (80, 150, 280, 400)),    # levý sloupec
        _lb("page_header", (80, 20, 300, 30)),
        _lb("text", (80, 450, 520, 500)),    # odstavec přes celou šířku
    ]
    out = XYCutReadingOrder().order(blocks, G)
    assert [(b.type.value, b.bbox.x0) for b in out] == [
        ("page_header", 80), ("title", 100), ("text", 80), ("text", 320), ("text", 80), ("page_footer", 250)]
    assert [b.order for b in out] == list(range(6))


def test_model_reading_order_keeps_model_order():
    from atr.adapters.reading_order.adapters import ModelReadingOrder

    out = ModelReadingOrder().order([_lb("text", (0, 500, 10, 510), 1), _lb("text", (0, 0, 10, 10), 0),
                                     _lb("text", (0, 200, 10, 210))], G)
    assert [b.bbox.y0 for b in out] == [0, 500, 200]


def test_crop_masks_neighbouring_blocks():
    import io

    from PIL import Image

    from atr.ports.types import PageImage
    from atr.render_utils import crop

    img = Image.new("L", (200, 200), 0)  # celá strana černá
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page = PageImage(1, 72, 200, 200, buf.getvalue())
    own, neighbour = BBox(10, 10, 100, 50), BBox(10, 45, 100, 90)
    out = Image.open(io.BytesIO(crop(page, own, padding_pt=10, mask=[neighbour]).png))
    # výřez začíná na (0, 0); vlastní blok zůstává černý, pás souseda pod ním je bílý
    assert out.getpixel((50, 30)) == 0
    assert out.getpixel((50, 55)) == 255
    assert out.getpixel((50, 47)) == 0  # překryv s vlastním blokem se nezabílí
