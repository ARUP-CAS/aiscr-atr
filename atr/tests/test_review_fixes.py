"""Regresní testy k nálezům z revize kódu (listopad 2026)."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from atr import config, registry, validation
from atr.adapters.layout.http import HttpLayoutDetector
from atr.adapters.layout_refiner.overlaps import OverlapRefiner
from atr.adapters.pdfium.adapters import PdfiumPageSource, PdfiumRenderer
from atr.adapters.pdfium.document import _user_units
from atr.adapters.recognizer.openai import OpenAIBlockRecognizer
from atr.adapters.text_layer_judge.rules import RulesTextLayerJudge
from atr.api import app as api
from atr.pipeline import Pipeline, PipelineError
from atr.ports.errors import DependencyError, ProcessingError
from atr.ports.types import (
    BBox,
    BlockType,
    DocumentInput,
    LayoutBlock,
    Line,
    PageGeometry,
    PageImage,
    TextLayer,
    TextLayerVerdict,
    Word,
)
from atr.tests.test_pipeline import DOC, doc, make_pipeline

registry.load_builtin_adapters()
ROOT = Path(__file__).resolve().parents[2]
G = PageGeometry(1, 595, 842, 0)


def _png(size=(100, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("L", size, 255).save(buf, format="PNG")
    return buf.getvalue()


def _client_with(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# --- 1. text mimo CropBox ---------------------------------------------------


@pytest.mark.parametrize("name", ["cropped", "cropped_rotated_90", "rotated_270", "user_unit_2", "born_digital"])
def test_every_sample_gives_valid_output(tmp_path_factory, name):
    """Celé potrubí s validací na vzorcích; text za hranicí CropBox nesmí shodit výstup."""
    pytest.importorskip("reportlab")
    sys.path.insert(0, str(ROOT / "samples"))
    import generate

    out_dir = tmp_path_factory.mktemp("s")
    generate.generate(out_dir)
    raw = {
        "ports": {
            "page_source": {"adapter": "pdfium"}, "renderer": {"adapter": "pdfium"},
            "text_layer": {"adapter": "pdfium"}, "text_layer_judge": {"adapter": "rules"},
            "router": {"adapter": "rules"},
            "layout": {"adapter": "fake", "params": {"blocks_by_page": {
                1: [{"type": "text", "bbox": [10, 10, 300, 300], "order": 0}]}}},
            "text_assigner": {"adapter": "overlap"}, "assembler": {"adapter": "docproc"},
        },
        "recognizers": {"text": {"adapter": "fake"}}, "recognize": {"text": "text"},
        "pipeline": {"render_dpi": 72, "min_text_coverage": 0},
    }
    cfg = config.parse(raw)
    out = Pipeline(cfg, config.build(cfg)).run(DocumentInput((out_dir / f"{name}.pdf").read_bytes(), name))
    assert validation.errors(out) == []
    page = out["pages"][0]
    for block in page["blocks"]:
        x0, y0, x1, y1 = block["bbox"]
        assert 0 <= x0 <= x1 <= page["width"] + 0.5 and 0 <= y0 <= y1 <= page["height"] + 0.5


# --- 2. služba neblokuje, limit velikosti ------------------------------------


def test_convert_runs_outside_event_loop(monkeypatch):
    seen = {}

    class Blocking:
        def run_with_paradata(self, document):
            try:
                asyncio.get_running_loop()
                seen["in_loop"] = True
            except RuntimeError:
                seen["in_loop"] = False
            return make_pipeline().run_with_paradata(document)

    monkeypatch.setattr(api, "get_pipeline", lambda: Blocking())
    r = TestClient(api.app).post("/convert", files={"file": ("d.json", json.dumps(DOC), "application/json")})
    assert r.status_code == 200
    assert seen["in_loop"] is False  # potrubí běží ve vlákně, smyčka událostí zůstává volná


def test_upload_limit(monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_MB", 0)
    r = TestClient(api.app).post("/convert", files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 413
    assert r.json()["reason"] == "payload_too_large"


# --- 3. překryv různých typů neukrojí větší blok ------------------------------


def test_deep_overlap_of_different_types_is_left_alone():
    text = LayoutBlock(BlockType.TEXT, BBox(50, 100, 500, 400), 0.9)
    note = LayoutBlock(BlockType.HANDWRITING, BBox(300, 200, 600, 220), 0.9)
    out = OverlapRefiner().refine([text, note], G)
    assert [b.bbox for b in out] == [text.bbox, note.bbox]


def test_thin_edge_overlap_is_still_split():
    a = LayoutBlock(BlockType.TEXT, BBox(50, 100, 500, 200), 0.9)
    b = LayoutBlock(BlockType.TABLE, BBox(50, 195, 500, 400), 0.9)
    out = OverlapRefiner().refine([a, b], G)
    assert out[0].bbox.y1 == out[1].bbox.y0 == 197.5


# --- 4. trvalé a dočasné chyby služeb ----------------------------------------


@pytest.mark.parametrize("status,error", [(413, ProcessingError), (400, ProcessingError),
                                          (500, DependencyError), (503, DependencyError),
                                          (429, DependencyError)])
def test_layout_http_errors_are_classified(status, error):
    detector = HttpLayoutDetector("http://layout")
    detector.client = _client_with(lambda request: httpx.Response(status, text="chyba"))
    with pytest.raises(error):
        detector.detect(PageImage(1, 72, 100, 100, _png()), G)


def test_layout_rejection_is_permanent_422(monkeypatch):
    pipeline = make_pipeline()

    class Rejecting:
        def detect(self, image, geometry):
            raise ProcessingError("layout_rejected", "HTTP 413")

        def info(self):
            return pipeline.c["layout"].info()

    pipeline.c.adapters["layout"] = Rejecting()
    with pytest.raises(PipelineError) as err:
        pipeline.run(doc())
    assert err.value.reason == "layout_rejected" and err.value.pages == [1]
    monkeypatch.setattr(api, "get_pipeline", lambda: pipeline)
    r = TestClient(api.app).post("/convert", files={"file": ("d.json", json.dumps(DOC), "application/json")})
    assert r.status_code == 422 and r.json()["reason"] == "layout_rejected"


def test_large_page_is_rendered_with_lower_dpi():
    spec = {"pages": [{"width": 2384, "height": 3370}]}  # A0
    pipeline = make_pipeline()
    pipeline.config = config.PipelineConfig(**{**pipeline.config.__dict__, "render_dpi": 200,
                                               "max_page_pixels": 10_000_000})
    seen = []
    original = pipeline.c["renderer"].render
    pipeline.c["renderer"].render = lambda d, n, dpi: seen.append(dpi) or original(d, n, dpi)
    pipeline.run(DocumentInput(json.dumps(spec).encode(), "a0.json"))
    assert seen and all(dpi < 200 for dpi in seen)
    assert (2384 / 72 * seen[0]) * (3370 / 72 * seen[0]) <= 10_000_000 * 1.001


# --- 5. useknutý výstup modelu ------------------------------------------------


def _recognizer(finish: str) -> OpenAIBlockRecognizer:
    r = OpenAIBlockRecognizer("http://ocr/v1", "m", "OCR:")
    r.client = _client_with(lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": "text"}, "finish_reason": finish}]}))
    return r


def test_truncated_recognition_is_rejected():
    block = LayoutBlock(BlockType.TEXT, BBox(0, 0, 10, 10), 0.9)
    crop = PageImage(1, 72, 100, 100, _png())
    assert _recognizer("stop").recognize(crop, block).text == "text"
    with pytest.raises(ProcessingError) as err:
        _recognizer("length").recognize(crop, block)
    assert err.value.reason == "recognition_incomplete"

    pipeline = make_pipeline()
    pipeline.c.recognizers["text"] = _recognizer("length")
    with pytest.raises(PipelineError) as failed:
        pipeline.run(doc())
    assert failed.value.reason == "all_engines_failed"


# --- 6. pořadí čtení bez komponenty reading_order ------------------------------


def test_order_is_renumbered_without_reading_order_port():
    spec = {"pages": [{}]}
    raw = {
        "ports": {
            "page_source": {"adapter": "fake"}, "renderer": {"adapter": "fake"},
            "text_layer": {"adapter": "fake"}, "text_layer_judge": {"adapter": "rules"},
            "router": {"adapter": "rules"},
            "layout": {"adapter": "fake", "params": {"blocks_by_page": {1: [
                {"type": "text", "bbox": [50, 50, 500, 300], "order": 0},
                {"type": "text", "bbox": [60, 60, 200, 100], "order": 1},   # vnořený, refiner ho vypustí
                {"type": "text", "bbox": [50, 400, 500, 600], "order": 2}]}}},
            "layout_refiner": {"adapter": "overlaps"},
            "text_assigner": {"adapter": "overlap"}, "assembler": {"adapter": "docproc"},
        },
        "recognizers": {"text": {"adapter": "fake"}}, "recognize": {"text": "text"},
        "pipeline": {"render_dpi": 72},
    }
    cfg = config.parse(raw)
    out = Pipeline(cfg, config.build(cfg)).run(DocumentInput(json.dumps(spec).encode(), "x"))
    assert [b["readingOrder"] for b in out["pages"][0]["blocks"]] == [0, 1]


# --- 7. pořadí čtení v datové sadě ---------------------------------------------


def test_dataset_reading_order_follows_xycut_without_predictions():
    spec = importlib.util.spec_from_file_location("to_paddlex", ROOT / "training" / "layout" / "to_paddlex.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    anns = [{"bbox": [50, 500, 400, 50]}, {"bbox": [50, 0, 400, 50]}, {"bbox": [50, 200, 400, 50]}]
    module.order_annotations(anns, [], 600, 800, 0.3)
    assert [a["read_order"] for a in anns] == [2, 0, 1]


# --- 8. jiná písma než latinka -------------------------------------------------


def _layer(text: str) -> TextLayer:
    words = tuple(Word(t, BBox(i * 10, 0, i * 10 + 9, 10)) for i, t in enumerate(text.split()))
    return TextLayer(1, (Line(text, BBox(0, 0, 500, 10), words),))


@pytest.mark.parametrize("text", [
    "Археологические исследования поселения эпохи бронзы были продолжены в следующем сезоне раскопок.",
    "Η αρχαιολογική έρευνα του οικισμού της εποχής του χαλκού συνεχίστηκε και την επόμενη περίοδο.",
])
def test_clean_non_latin_text_is_digital(text):
    assert RulesTextLayerJudge().judge(_layer(text), G).verdict is TextLayerVerdict.DIGITAL


# --- 9. orientace z EXIF a různé DPI v osách ------------------------------------


def test_exif_orientation_is_applied():
    img = Image.new("RGB", (400, 200), "white")
    exif = img.getexif()
    exif[0x0112] = 6  # otočit o 90° po směru hodin
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif, dpi=(200, 200))
    doc_ = PdfiumPageSource().open(DocumentInput(buf.getvalue(), "foto.jpg"))
    try:
        g = doc_.pages[0]
        assert g.source_pixels == (200, 400)
        assert (g.width, g.height) == pytest.approx((72.0, 144.0))
        image = PdfiumRenderer().render(doc_, 1, 100)
        assert (image.width_px, image.height_px) == (100, 200)
    finally:
        doc_.close()


def test_non_square_dpi_gives_true_page_size():
    buf = io.BytesIO()
    Image.new("L", (1728, 1078), 255).save(buf, format="TIFF", dpi=(204, 98))  # fax, A4
    doc_ = PdfiumPageSource().open(DocumentInput(buf.getvalue(), "fax.tif"))
    try:
        g = doc_.pages[0]
        assert (g.width, g.height) == pytest.approx((1728 * 72 / 204, 1078 * 72 / 98), rel=1e-3)
        image = PdfiumRenderer().render(doc_, 1, 72)
        assert (image.width_px, image.height_px) == pytest.approx((g.width, g.height), abs=1)
    finally:
        doc_.close()


# --- 10. paradata: jedinečná @id -------------------------------------------------


def test_paradata_instrument_ids_are_unique():
    _, para = make_pipeline().run_with_paradata(doc())
    ids = [i["@id"] for i in para["@graph"][0]["instrument"]]
    assert len(ids) == len(set(ids))
    assert sum(i["additionalType"] == "recognizer" for i in para["@graph"][0]["instrument"]) == 2


# --- drobnosti ---------------------------------------------------------------------


def test_recognizer_is_not_a_port():
    raw = {"ports": {**{p: {"adapter": "fake"} for p in config.REQUIRED_PORTS}, "recognizer": {"adapter": "fake"}}}
    with pytest.raises(config.ConfigError, match="recognizer"):
        config.parse(raw)


def test_user_unit_reader_survives_unparsable_pdf():
    assert _user_units(b"%PDF-1.7 /UserUnit poskozeno", 3) == [1.0, 1.0, 1.0]


# --- ruční písmo se jen označuje ------------------------------------------------


def test_handwriting_is_marked_but_not_recognized():
    """Blok handwriting zůstane ve výstupu s polohou a typem, bez textu a bez volání OCR,
    a to i když leží celý uvnitř odstavce."""
    spec = {"pages": [{}]}
    raw = {
        "ports": {
            "page_source": {"adapter": "fake"}, "renderer": {"adapter": "fake"},
            "text_layer": {"adapter": "fake"}, "text_layer_judge": {"adapter": "rules"},
            "router": {"adapter": "rules"},
            "layout": {"adapter": "fake", "params": {"blocks_by_page": {1: [
                {"type": "text", "bbox": [50, 50, 500, 300], "order": 0},
                {"type": "handwriting", "bbox": [300, 100, 480, 140], "order": 1}]}}},
            "layout_refiner": {"adapter": "overlaps"}, "reading_order": {"adapter": "model"},
            "text_assigner": {"adapter": "overlap"}, "assembler": {"adapter": "docproc"},
        },
        "recognizers": {"text": {"adapter": "fake"}}, "recognize": {"text": "text"},
        "pipeline": {"render_dpi": 72},
    }
    cfg = config.parse(raw)
    pipeline = Pipeline(cfg, config.build(cfg))
    out = pipeline.run(DocumentInput(json.dumps(spec).encode(), "x"))
    blocks = out["pages"][0]["blocks"]
    assert [(b["type"], b["content"]) for b in blocks] == [("text", "rozpoznaný text"), ("handwriting", "")]
    assert blocks[1]["bbox"] == [300, 100, 480, 140]
    assert [t for _, t in pipeline.c.recognizers["text"].calls] == [BlockType.TEXT]
    assert validation.errors(out) == []


def test_repository_config_does_not_recognize_handwriting():
    cfg = config.load(ROOT / "config" / "pipeline.yaml")
    assert BlockType.HANDWRITING not in cfg.recognize
    assert BlockType.FIGURE not in cfg.recognize
