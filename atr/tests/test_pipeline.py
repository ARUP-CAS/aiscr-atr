"""Testy toku potrubí s adaptéry `fake` (bez PDF, bez GPU)."""

from __future__ import annotations

import json

import pytest

from atr import config, registry
from atr.pipeline import Pipeline, PipelineError
from atr.ports.types import DocumentInput

registry.load_builtin_adapters()

DIGITAL_TEXT = "Archeologický výzkum v roce 1987 odkryl sídliště z mladší doby bronzové."

DOC = {
    "pages": [
        # 1: born-digital
        {"text_lines": [{"text": DIGITAL_TEXT, "bbox": [72, 100, 520, 115]}]},
        # 2: sken (bez textové vrstvy)
        {},
        # 3: prázdná
        {},
        # 4: vadná textová vrstva
        {"text_lines": [{"text": "��� �� x ���� ���",
                         "bbox": [72, 100, 520, 115]}]},
    ]
}

LAYOUT = {
    1: [{"type": "text", "bbox": [70, 95, 525, 120], "order": 0}],
    2: [{"type": "heading", "bbox": [70, 60, 400, 80], "order": 0},
        {"type": "table", "bbox": [70, 100, 520, 300], "order": 1},
        {"type": "figure", "bbox": [70, 320, 520, 600], "order": 2}],
    4: [{"type": "text", "bbox": [70, 95, 525, 120], "order": 0}],
}


def make_pipeline(recognizer_params: dict | None = None, extra_ports: dict | None = None) -> Pipeline:
    raw = {
        "ports": {
            "page_source": {"adapter": "fake"},
            "renderer": {"adapter": "fake"},
            "text_layer": {"adapter": "fake"},
            "text_layer_judge": {"adapter": "rules"},
            "router": {"adapter": "rules"},
            "layout": {"adapter": "fake", "params": {"blocks_by_page": LAYOUT}},
            "text_assigner": {"adapter": "overlap"},
            "assembler": {"adapter": "docproc"},
            "language": {"adapter": "fake"},
            **(extra_ports or {}),
        },
        "recognizers": {
            "text": {"adapter": "fake", "params": recognizer_params or {}},
            "table": {"adapter": "fake", "params": {"text": "<table></table>"}},
        },
        "recognize": {"text": "text", "heading": "text", "table": "table"},
        "pipeline": {"render_dpi": 72},
    }
    cfg = config.parse(raw)
    return Pipeline(cfg, config.build(cfg))


def doc(forced: dict[int, str] | None = None) -> DocumentInput:
    return DocumentInput(json.dumps(DOC).encode(), "test.json", forced_pages=forced or {})


def routes(out: dict) -> list[tuple[int, str]]:
    return [(p["page"], p["route"]) for p in out["metadata"]["routing"]["pages"]]


def test_every_page_is_in_routing():
    out = make_pipeline().run(doc())
    assert out["pageCount"] == 4
    assert routes(out) == [(1, "digital"), (2, "ocr"), (3, "empty"), (4, "ocr")]


def test_empty_page_has_no_pages_entry():
    out = make_pipeline().run(doc())
    assert [p["pageNumber"] for p in out["pages"]] == [1, 2, 4]


def test_digital_page_takes_text_layer_with_measured_lines():
    out = make_pipeline().run(doc())
    block = out["pages"][0]["blocks"][0]
    assert block["content"] == DIGITAL_TEXT
    assert block["lines"][0]["position"] == "measured"
    assert block["language"] == "cs"


def test_scanned_page_is_recognized_per_block_by_type():
    pipeline = make_pipeline()
    out = pipeline.run(doc())
    blocks = out["pages"][1]["blocks"]
    assert [b["type"] for b in blocks] == ["heading", "table", "figure"]
    assert blocks[0]["content"] == "rozpoznaný text"
    assert blocks[0]["lines"][0]["position"] == "derived"
    assert blocks[1]["content"] == "<table></table>"
    assert blocks[2]["content"] == ""  # obrázek se nerozpoznává
    assert pipeline.c.recognizers["table"].calls == [(2, pipeline.c.recognizers["table"].calls[0][1])]


def test_garbled_text_layer_goes_to_ocr():
    out = make_pipeline().run(doc())
    entry = out["metadata"]["routing"]["pages"][3]
    assert entry["text_layer"] == "garbled"
    assert out["pages"][2]["blocks"][0]["content"] == "rozpoznaný text"


def test_forced_page_is_recognized_and_marked():
    out = make_pipeline().run(doc({1: "broken_text_layer"}))
    entry = out["metadata"]["routing"]["pages"][0]
    assert entry["route"] == "forced"
    assert entry["forced_reason"] == "broken_text_layer"
    assert out["pages"][0]["blocks"][0]["content"] == "rozpoznaný text"


def test_forced_page_outside_document_is_an_error():
    with pytest.raises(PipelineError) as err:
        make_pipeline().run(doc({9: "x"}))
    assert err.value.reason == "invalid_forced_pages"


def test_recognizer_failure_fails_the_job():
    with pytest.raises(PipelineError) as err:
        make_pipeline({"fail": True}).run(doc())
    assert err.value.reason == "all_engines_failed"
    assert err.value.pages == [2]


def test_page_classifier_is_optional_and_reported():
    out = make_pipeline(extra_ports={"page_classifier": {"adapter": "fake",
                                                         "params": {"labels": ["text", "photo"]}}}).run(doc())
    assert out["metadata"]["routing"]["pages"][1]["label"] == "photo"


def test_no_run_data_in_output():
    out = make_pipeline().run(doc())
    for key in ("pipeline", "pipelineVersion", "processedAt", "sourceFile", "sourceHash", "warnings"):
        assert key not in out


def test_components_are_reported_for_paradata():
    pipeline = make_pipeline()
    result = pipeline.process(doc())
    ports = {c.port for c in result.components}
    assert {"layout", "router", "assembler", "recognizer"} <= ports


def test_unavailable_recognizer_is_not_a_recognition_failure():
    from atr.adapters.recognizer.openai import OpenAIBlockRecognizer
    from atr.ports.errors import DependencyError

    pipeline = make_pipeline()
    pipeline.c.recognizers["text"] = OpenAIBlockRecognizer("http://127.0.0.1:9/v1", "m", "OCR:", timeout_s=2)
    with pytest.raises(DependencyError):
        pipeline.run(doc())


def test_paradata_describe_run_and_hash_output():
    import hashlib

    from atr.paradata import canonical_json

    output, para = make_pipeline().run_with_paradata(doc({4: "needs_atr"}))
    action = para["@graph"][0]
    assert action["@type"] == "CreateAction"
    assert action["result"][0]["sha512"] == hashlib.sha512(canonical_json(output)).hexdigest()
    assert action["wfrun:summary"]["routes"] == {"digital": 1, "ocr": 1, "empty": 1, "forced": 1}
    assert action["wfrun:summary"]["forced_pages"] == {"4": "needs_atr"}
    ports = {i["additionalType"] for i in action["instrument"]}
    assert {"layout", "router", "assembler"} <= ports
    assert action["startTime"] and action["endTime"]


def test_document_language_from_blocks():
    out = make_pipeline().run(doc())
    assert out["documentLanguage"] == "cs"


def test_incomplete_text_layer_goes_to_ocr():
    """Text jen v jednom ze dvou stejně velkých textových bloků → pokrytí 0,5 → OCR."""
    spec = {"pages": [{"text_lines": [{"text": DIGITAL_TEXT, "bbox": [72, 100, 520, 115]}]}]}
    raw = {
        "ports": {
            "page_source": {"adapter": "fake"}, "renderer": {"adapter": "fake"},
            "text_layer": {"adapter": "fake"}, "text_layer_judge": {"adapter": "rules"},
            "router": {"adapter": "rules"},
            "layout": {"adapter": "fake", "params": {"blocks_by_page": {1: [
                {"type": "text", "bbox": [70, 95, 525, 120], "order": 0},
                {"type": "text", "bbox": [70, 300, 525, 325], "order": 1}]}}},
            "text_assigner": {"adapter": "overlap"}, "assembler": {"adapter": "docproc"},
        },
        "recognizers": {"text": {"adapter": "fake"}},
        "recognize": {"text": "text"},
        "pipeline": {"render_dpi": 72},
    }
    cfg = config.parse(raw)
    out = Pipeline(cfg, config.build(cfg)).run(DocumentInput(json.dumps(spec).encode(), "x.json"))
    entry = out["metadata"]["routing"]["pages"][0]
    assert entry["route"] == "ocr"
    assert entry["reason"].startswith("text_layer_incomplete")
    assert [b["content"] for b in out["pages"][0]["blocks"]] == ["rozpoznaný text"] * 2


def test_output_conforms_to_schema_draft():
    from atr import validation

    for forced in ({}, {4: "needs_atr"}):
        out = make_pipeline().run(doc(forced))
        assert validation.errors(out) == []


def test_validation_catches_missing_page_in_routing():
    from atr import validation

    out = make_pipeline().run(doc())
    out["metadata"]["routing"]["pages"].pop(1)
    assert any("routing" in e for e in validation.errors(out))
    out = make_pipeline().run(doc())
    out["pages"][0]["blocks"][0]["bbox"] = [0, 0, 9999, 10]
    assert any("mimo stranu" in e for e in validation.errors(out))
