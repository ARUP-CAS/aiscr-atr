"""Regresní testy k nálezům z druhé revize kódu (listopad 2026)."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from atr.adapters.layout import http as layout_http
from atr.adapters.layout.http import HttpLayoutDetector
from atr.adapters.pdfium import document as pdfium_document
from atr.adapters.pdfium.adapters import PdfiumPageSource, PdfiumRenderer, PdfiumTextLayerReader
from atr.adapters.recognizer.openai import OpenAIBlockRecognizer
from atr.adapters.text_layer_judge.rules import RulesTextLayerJudge
from atr.api import app as api
from atr.ports.errors import InputError, ProcessingError
from atr.ports.types import (
    BBox,
    BlockType,
    DocumentInput,
    LayoutBlock,
    PageGeometry,
    PageImage,
    TextLayerVerdict,
)
from atr.tests.test_pipeline import DOC, make_pipeline

ROOT = Path(__file__).resolve().parents[2]


def _open(data: bytes, name: str = "x"):
    return PdfiumPageSource().open(DocumentInput(data, name))


def _render(data: bytes, dpi: float = 72) -> Image.Image:
    doc = _open(data)
    try:
        return Image.open(io.BytesIO(PdfiumRenderer().render(doc, 1, dpi).png))
    finally:
        doc.close()


def _png(size=(100, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("L", size, 255).save(buf, format="PNG")
    return buf.getvalue()


# --- 1. bitová hloubka a průhlednost ---------------------------------------------


def test_16bit_grayscale_scan_is_not_rendered_white():
    img = Image.new("I", (200, 100), 60000)          # světlé pozadí
    img.paste(2000, (50, 30, 150, 70))               # tmavý obdélník („text“)
    img = img.convert("I;16")
    assert img.getextrema() == (2000, 60000)
    buf = io.BytesIO()
    img.save(buf, format="TIFF", dpi=(300, 300))
    out = _render(buf.getvalue(), 300).convert("L")
    assert out.getpixel((100, 50)) < 30
    assert out.getpixel((10, 10)) > 200


def test_transparent_png_gets_white_background():
    img = Image.new("RGBA", (200, 100), (0, 0, 0, 0))  # průhledné pozadí
    ImageDraw.Draw(img).rectangle((50, 30, 150, 70), fill=(0, 0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=(300, 300))
    out = _render(buf.getvalue(), 300).convert("L")
    assert out.getpixel((10, 10)) == 255
    assert out.getpixel((100, 50)) == 0


# --- 2. sken zabalený ve Form XObject ----------------------------------------------


def test_scan_inside_form_xobject_counts_as_image_coverage():
    pytest.importorskip("reportlab")
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    scan = io.BytesIO()
    Image.new("L", (600, 800), 200).save(scan, format="PNG")
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(595, 842))
    c.beginForm("sken")
    c.drawImage(ImageReader(io.BytesIO(scan.getvalue())), 0, 0, 595, 842)
    c.endForm()
    c.doForm("sken")
    text = c.beginText(72, 700)
    text.setTextRenderMode(3)
    for _ in range(10):
        text.textLine("Nalezova zprava o vyzkumu sidliste z doby bronzove a nalezech keramiky.")
    c.drawText(text)
    c.showPage()
    c.save()
    doc = _open(buf.getvalue())
    try:
        layer = PdfiumTextLayerReader().read(doc, 1)
        verdict = RulesTextLayerJudge().judge(layer, doc.pages[0])
    finally:
        doc.close()
    assert layer.image_coverage > 0.9
    assert verdict.verdict is TextLayerVerdict.OCR


# --- 3. bloky rozvržení se oříznou na stranu ------------------------------------------


def test_layout_blocks_are_clamped_to_page():
    detector = HttpLayoutDetector("http://layout", label_map={"text": "text"})
    detector.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
        "blocks": [{"label": "text", "bbox_px": [-3, 10, 105, 101], "confidence": 0.9, "order": 0},
                   {"label": "text", "bbox_px": [120, 10, 130, 20], "confidence": 0.9, "order": 1}]})))
    geometry = PageGeometry(1, 100, 100, 0)
    blocks = detector.detect(PageImage(1, 72, 101, 101, _png()), geometry)
    assert len(blocks) == 1  # blok celý mimo stranu vypadl
    b = blocks[0].bbox
    assert (b.x0, b.y0, b.x1, b.y1) == (0, 10, 100, 100)


def test_layout_info_is_refreshed(monkeypatch):
    versions = iter(["v1", "v2"])
    detector = HttpLayoutDetector("http://layout")
    detector.client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"model": "m", "version": next(versions)})))
    assert detector.info().version == "v1"
    assert detector.info().version == "v1"  # v rámci TTL z mezipaměti
    monkeypatch.setattr(layout_http, "INFO_TTL_S", 0.0)
    assert detector.info().version == "v2"


# --- 4. příliš velký obrázek je chyba vstupu ------------------------------------------


def test_decompression_bomb_is_input_error(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    buf = io.BytesIO()
    Image.new("1", (100, 100), 1).save(buf, format="PNG")  # 10 000 px > 2 × 1000
    with pytest.raises(InputError) as err:
        pdfium_document.ImageDocument(buf.getvalue(), 300)
    assert err.value.reason == "image_too_large"


# --- 5. poškozený strom stran -----------------------------------------------------------


def test_broken_page_tree_is_input_error():
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(595, 842))
    c.drawString(72, 700, "strana")
    c.showPage()
    c.save()
    data = buf.getvalue()
    assert b"/Count 1" in data
    broken = data.replace(b"/Count 1", b"/Count 3")  # strom stran slibuje víc stran, než má
    with pytest.raises(InputError) as err:
        _open(broken)
    assert err.value.reason == "unreadable_document"


# --- 6. strany se načítají postupně ------------------------------------------------------


def test_image_frames_are_decoded_on_demand():
    frames = [Image.new("L", (60, 40), value) for value in (10, 120, 240)]
    buf = io.BytesIO()
    frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:], dpi=(300, 300))
    doc = _open(buf.getvalue())
    try:
        assert len(doc.pages) == 3
        assert not hasattr(doc, "frames")  # žádný seznam dekódovaných snímků
        renderer = PdfiumRenderer()
        got = [Image.open(io.BytesIO(renderer.render(doc, n, 300).png)).getpixel((5, 5)) for n in (3, 1, 2)]
    finally:
        doc.close()
    assert got == [240, 10, 120]


def test_pdf_keeps_only_one_page_loaded(tmp_path_factory):
    pytest.importorskip("reportlab")
    sys.path.insert(0, str(ROOT / "samples"))
    import generate

    out = tmp_path_factory.mktemp("s")
    generate.generate(out)
    doc = _open((out / "mixed.pdf").read_bytes())
    try:
        assert doc._current is None  # při otevření se čte jen geometrie
        first = doc.page(1)
        assert doc.page(1) is first
        doc.page(3)
        assert doc._current[0] == 3
        assert PdfiumTextLayerReader().read(doc, 1).lines  # návrat na dřívější stranu funguje
    finally:
        doc.close()


# --- 7. useknutý blok se jednou zkusí znovu s větším limitem ---------------------------------


def test_truncated_block_is_retried_with_more_tokens():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        max_tokens = json.loads(request.content)["max_tokens"]
        seen.append(max_tokens)
        finish = "length" if max_tokens <= 100 else "stop"
        return httpx.Response(200, json={"choices": [{"message": {"content": "tabulka"}, "finish_reason": finish}]})

    recognizer = OpenAIBlockRecognizer("http://ocr/v1", "m", "Table Recognition:", max_tokens=100)
    recognizer.client = httpx.Client(transport=httpx.MockTransport(handler))
    block = LayoutBlock(BlockType.TABLE, BBox(0, 0, 10, 10), 0.9)
    assert recognizer.recognize(PageImage(1, 72, 100, 100, _png()), block).text == "tabulka"
    assert seen == [100, 200]

    always = OpenAIBlockRecognizer("http://ocr/v1", "m", "OCR:", max_tokens=100)
    always.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json={"choices": [{"message": {"content": "x"}, "finish_reason": "length"}]})))
    with pytest.raises(ProcessingError):
        always.recognize(PageImage(1, 72, 100, 100, _png()), block)


# --- 8. UserUnit v komprimovaných objektech ---------------------------------------------------


def test_user_unit_is_read_when_objects_are_compressed(monkeypatch):
    import pypdf

    calls = []

    class FakeReader:
        def __init__(self, stream):
            calls.append(True)
            self.pages = [{"/UserUnit": 2}, {}]

    monkeypatch.setattr(pypdf, "PdfReader", FakeReader)
    # klíč není v čitelných bajtech, ale soubor má komprimované objekty → pypdf se zeptáme
    assert pdfium_document._user_units(b"%PDF-1.7 ... /Type /ObjStm ...", 2) == [2.0, 1.0]
    assert calls
    calls.clear()
    assert pdfium_document._user_units(b"%PDF-1.4 bez objektovych proudu", 2) == [1.0, 1.0]
    assert not calls  # obyčejné PDF bez klíče pypdf nevolá


# --- 9. příkazová řádka vydává bajty, ke kterým patří otisk -------------------------------------


def test_cli_output_matches_paradata_hash(tmp_path, capfdbinary):
    from atr.__main__ import main

    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(json.dumps({
        "ports": {**{p: {"adapter": "fake"} for p in ("page_source", "renderer", "text_layer", "layout")},
                  "text_layer_judge": {"adapter": "rules"}, "router": {"adapter": "rules"},
                  "text_assigner": {"adapter": "overlap"}, "assembler": {"adapter": "docproc"}},
        "recognizers": {"text": {"adapter": "fake"}}, "recognize": {"text": "text"},
        "pipeline": {"render_dpi": 72},
    }), encoding="utf-8")  # JSON je platný YAML
    source = tmp_path / "doc.json"
    source.write_text(json.dumps(DOC), encoding="utf-8")
    para_path = tmp_path / "para.json"
    assert main(["convert", str(source), "--config", str(cfg), "--paradata", str(para_path)]) == 0
    out = capfdbinary.readouterr().out
    result = json.loads(para_path.read_text(encoding="utf-8"))["@graph"][0]["result"][0]
    assert hashlib.sha512(out).hexdigest() == result["sha512"]
    assert len(out) == result["contentSize"]


# --- 10. potrubí se sestaví při startu ---------------------------------------------------------


def test_pipeline_is_built_at_startup(monkeypatch):
    built = []
    pipeline = make_pipeline()
    monkeypatch.setattr(api, "get_pipeline", lambda: built.append(True) or pipeline)
    with TestClient(api.app) as client:
        assert built  # ještě před prvním požadavkem
        assert client.get("/health").json() == {"status": "ok"}


def test_compose_forwards_service_urls():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for name in ("LAYOUT_URL", "OCR_TEXT_URL", "OCR_ALT_URL"):
        assert f"{name}: ${{{name}:-" in compose
