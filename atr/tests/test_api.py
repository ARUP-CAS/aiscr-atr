"""API s konfigurací z repozitáře; kontejnery modelů neběží."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from atr.adapters.layout.http import HttpLayoutDetector
from atr.api import app as api
from atr.ports.errors import DependencyError
from atr.ports.types import PageGeometry, PageImage

ROOT = Path(__file__).resolve().parents[2]
DEAD_URL = "http://127.0.0.1:9"  # port discard: spojení se odmítne


@pytest.fixture()
def client(monkeypatch, tmp_path):
    cfg = (ROOT / "config" / "pipeline.yaml").read_text(encoding="utf-8")
    cfg = cfg.replace("${LAYOUT_URL:-http://layout:8000}", DEAD_URL)
    path = tmp_path / "pipeline.yaml"
    path.write_text(cfg, encoding="utf-8")
    monkeypatch.setenv("ATR_CONFIG", str(path))
    api.get_pipeline.cache_clear()
    yield TestClient(api.app)
    api.get_pipeline.cache_clear()


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("L", (100, 100), 255).save(buf, format="PNG")
    return buf.getvalue()


def test_health_and_info_work_without_model_services(client):
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/info").json()
    assert info["contract"]["name"] == "atr/docproc-json"
    layout = next(c for c in info["components"] if c["port"] == "layout")
    assert layout["name"] == "unreachable"


def test_unavailable_layout_is_503(client):
    r = client.post("/convert", files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 503
    assert r.json()["reason"] == "service_unavailable"


def test_unsupported_input_is_422(client):
    r = client.post("/convert", files={"file": ("a.bin", b"nic", "application/octet-stream")})
    assert r.status_code == 422
    assert r.json()["reason"] == "unsupported_format"


def test_forced_pages_must_be_json(client):
    r = client.post("/convert", files={"file": ("a.png", _png(), "image/png")},
                    data={"forced_pages": "3"})
    assert r.status_code == 400
    r = client.post("/convert", files={"file": ("a.png", _png(), "image/png")},
                    data={"forced_pages": json.dumps({"7": "x"})})
    assert r.status_code == 400
    assert r.json()["pages"] == [7]


def test_layout_adapter_reports_dependency_error():
    detector = HttpLayoutDetector(DEAD_URL, timeout_s=2)
    image = PageImage(1, 72, 100, 100, _png())
    with pytest.raises(DependencyError):
        detector.detect(image, PageGeometry(1, 100, 100, 0))


def test_layout_adapter_converts_pixels_to_points_and_maps_labels():
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict"
        return httpx.Response(200, json={"blocks": [
            {"label": "paragraph_title", "bbox_px": [200, 100, 400, 150], "confidence": 0.9, "order": 0},
            {"label": "neznama_trida", "bbox_px": [0, 0, 10, 10], "confidence": 0.8, "order": 1},
            {"label": "text", "bbox_px": [0, 0, 10, 10], "confidence": 0.1, "order": 2},
        ]})

    detector = HttpLayoutDetector("http://layout", label_map={"paragraph_title": "heading"}, min_confidence=0.3)
    detector.client = httpx.Client(transport=httpx.MockTransport(handler))
    blocks = detector.detect(PageImage(1, 200, 100, 100, _png()), PageGeometry(1, 36, 36, 0))
    assert [b.type.value for b in blocks] == ["heading", "other"]  # nízká jistota vypadla
    assert (blocks[0].bbox.x0, blocks[0].bbox.y1) == pytest.approx((72.0, 54.0))  # px × 72/200
    assert blocks[0].source_label == "paragraph_title"
