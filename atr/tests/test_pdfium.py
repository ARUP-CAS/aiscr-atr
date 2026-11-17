"""Adaptéry `pdfium` na syntetické sadě se známou geometrií (samples/generate.py)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from atr import config, registry
from atr.adapters.pdfium.adapters import PdfiumPageSource, PdfiumRenderer, PdfiumTextLayerReader
from atr.pipeline import Pipeline
from atr.ports.errors import InputError
from atr.ports.types import DocumentInput

pytest.importorskip("reportlab")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "samples"))
import generate

registry.load_builtin_adapters()

TOL = 3.0  # bodů; volné rámečky znaků jsou o něco větší než odhad z metrik písma
PDF_SAMPLES = ["born_digital", "rotated_90", "rotated_180", "rotated_270", "cropped",
               "cropped_rotated_90", "user_unit_2"]


@pytest.fixture(scope="session")
def samples(tmp_path_factory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("samples")
    return out, generate.generate(out)


def _open(path: Path):
    return PdfiumPageSource().open(DocumentInput(path.read_bytes(), path.name))


def _marker(layer) -> tuple[float, float, float, float]:
    word = next(w for w in layer.words if w.text == generate.MARKER)
    return word.bbox.x0, word.bbox.y0, word.bbox.x1, word.bbox.y1


@pytest.mark.parametrize("name", PDF_SAMPLES)
def test_geometry_matches_displayed_page(samples, name):
    root, expected = samples
    doc = _open(root / f"{name}.pdf")
    try:
        g = doc.pages[0]
        exp = expected[name]["page"]
        assert (g.width, g.height, g.rotation, g.user_unit) == pytest.approx(
            (exp["width"], exp["height"], exp["rotation"], exp["user_unit"]))
    finally:
        doc.close()


@pytest.mark.parametrize("name", PDF_SAMPLES)
def test_word_position_on_displayed_page(samples, name):
    root, expected = samples
    doc = _open(root / f"{name}.pdf")
    try:
        got = _marker(PdfiumTextLayerReader().read(doc, 1))
    finally:
        doc.close()
    exp = expected[name]["marker_bbox"]
    uu = expected[name]["page"]["user_unit"]
    assert got == pytest.approx(exp, abs=TOL * uu)


@pytest.mark.parametrize("name", ["born_digital", "rotated_90", "cropped_rotated_90", "user_unit_2"])
def test_render_and_text_share_coordinates(samples, name):
    """Na vykreslené straně je pod polohou značky inkoust, mimo ni (vpravo) ne."""
    root, _ = samples
    doc = _open(root / f"{name}.pdf")
    try:
        layer = PdfiumTextLayerReader().read(doc, 1)
        image = PdfiumRenderer().render(doc, 1, 72)
        g = doc.pages[0]
    finally:
        doc.close()
    assert (image.width_px, image.height_px) == (round(g.width), round(g.height))
    img = Image.open(io.BytesIO(image.png)).convert("L")
    x0, y0, x1, y1 = _marker(layer)
    inside = img.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))
    assert inside.getextrema()[0] < 128
    blank_x = x1 + 2 * (x1 - x0) if x1 + 2 * (x1 - x0) < g.width - (x1 - x0) else x0 - 2 * (x1 - x0)
    outside = img.crop((int(blank_x), int(y0), int(blank_x + x1 - x0), int(y1) + 1))
    assert outside.getextrema()[0] > 200


def test_czech_text_layer(samples):
    root, expected = samples
    if not expected["_font"]["czech"]:
        pytest.skip("bez písma s českými znaky (nastav ATR_TEST_FONT)")
    doc = _open(root / "born_digital.pdf")
    try:
        layer = PdfiumTextLayerReader().read(doc, 1)
    finally:
        doc.close()
    assert "Žluťoučký kůň úpěl ďábelské ódy." in layer.text
    assert layer.unmapped_chars == 0
    assert layer.lines[0].text == "Nálezová zpráva"


def test_scan_and_mixed_text_layers(samples):
    root, _ = samples
    doc = _open(root / "mixed.pdf")
    try:
        reader = PdfiumTextLayerReader()
        assert len(doc.pages) == 3
        has_text = [bool(reader.read(doc, n).lines) for n in (1, 2, 3)]
    finally:
        doc.close()
    assert has_text == [True, False, False]


@pytest.mark.parametrize("name,pages", [("scan_300dpi.png", 1), ("two_pages.tiff", 2)])
def test_image_input(samples, name, pages):
    root, expected = samples
    doc = _open(root / name)
    try:
        assert len(doc.pages) == pages
        g = doc.pages[0]
        assert g.source_pixels == tuple(expected[name]["source_pixels"])
        assert PdfiumTextLayerReader().read(doc, 1).lines == ()
        image = PdfiumRenderer().render(doc, 1, 150)
        assert (image.width_px, image.height_px) == pytest.approx(
            (g.width * 150 / 72, g.height * 150 / 72), abs=1)
    finally:
        doc.close()
    if name == "scan_300dpi.png":
        assert (g.width, g.height) == pytest.approx(
            (expected[name]["page"]["width"], expected[name]["page"]["height"]), abs=0.01)  # PNG ukládá DPI zaokrouhleně


def test_unreadable_input_is_reported():
    with pytest.raises(InputError) as err:
        PdfiumPageSource().open(DocumentInput(b"tohle neni dokument", "x.bin"))
    assert err.value.reason == "unsupported_format"
    with pytest.raises(InputError) as err:
        PdfiumPageSource().open(DocumentInput(b"%PDF-1.7\nposkozeno", "x.pdf"))
    assert err.value.reason == "unreadable_document"


def test_pipeline_on_mixed_pdf(samples):
    """Smíšené PDF projde potrubím s reálnou čtečkou: text, sken, prázdná strana."""
    root, _ = samples
    raw = {
        "ports": {
            "page_source": {"adapter": "pdfium"},
            "renderer": {"adapter": "pdfium"},
            "text_layer": {"adapter": "pdfium"},
            "text_layer_judge": {"adapter": "rules"},
            "router": {"adapter": "rules"},
            "layout": {"adapter": "fake", "params": {"blocks_by_page": {
                1: [{"type": "text", "bbox": [0, 0, 595, 842], "order": 0}],
                2: [{"type": "text", "bbox": [50, 50, 500, 200], "order": 0}],
            }}},
            "text_assigner": {"adapter": "overlap"},
            "assembler": {"adapter": "docproc"},
        },
        "recognizers": {"text": {"adapter": "fake"}},
        "recognize": {"text": "text"},
        "pipeline": {"render_dpi": 72},
    }
    cfg = config.parse(raw)
    out = Pipeline(cfg, config.build(cfg)).run(
        DocumentInput((root / "mixed.pdf").read_bytes(), "mixed.pdf"))
    routes = [(p["page"], p["route"]) for p in out["metadata"]["routing"]["pages"]]
    assert routes == [(1, "digital"), (2, "ocr"), (3, "empty")]
    assert generate.MARKER in out["pages"][0]["blocks"][0]["content"]
    assert out["pages"][1]["blocks"][0]["content"] == "rozpoznaný text"
    json.dumps(out)  # výstup jde serializovat


@pytest.mark.parametrize("name", ["old_ocr", "garbled_mojibake", "garbled_spaced", "garbled_glued",
                                  "garbled_nonsense", "born_digital", "rotated_90", "scan"])
def test_text_layer_verdicts(samples, name):
    """Starší OCR vrstva → ocr, nekvalitní text → garbled, dobrý text → digital, sken → none."""
    from atr.adapters.text_layer_judge.rules import RulesTextLayerJudge

    root, expected = samples
    if name not in expected:
        pytest.skip("vzorek vyžaduje písmo s češtinou")
    doc = _open(root / f"{name}.pdf")
    try:
        layer = PdfiumTextLayerReader().read(doc, 1)
        verdict = RulesTextLayerJudge().judge(layer, doc.pages[0])
    finally:
        doc.close()
    want = expected[name].get("verdict") or {"scan": "none"}.get(name, "digital")
    assert verdict.verdict.value == want, verdict.reason
