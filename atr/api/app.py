"""HTTP rozhraní služby: /convert, /info, /health.

Chybový model jako ostatní služby ATRIUM: {"status", "reason", "detail"}.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from atr import __version__, config, registry
from atr.adapters.assembler.docproc import CONTRACT_VERSION
from atr.paradata import canonical_json
from atr.pipeline import Pipeline, PipelineError
from atr.ports.errors import DependencyError, InputError
from atr.ports.types import DocumentInput

app = FastAPI(
    title="aiscr-atr",
    version=__version__,
    description=(
        "Otevřená služba ATR: z dokumentu (PDF, sken, obrázek) vytvoří `atr/docproc-json` "
        "(rozvržení, bloky s typem, polohou, pořadím čtení a textem; routing každé strany). "
        "Chyby mají tvar `{status, reason, detail}`: 400 neplatný požadavek, 422 nečitelný vstup "
        "nebo selhání rozpoznání (`all_engines_failed`), 503 nedostupná služba modelu (zopakovat), "
        "500 neplatný výstup."
    ),
    license_info={"name": "MIT"},
)


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    registry.load_builtin_adapters()
    cfg = config.load(os.environ.get("ATR_CONFIG", "config/pipeline.yaml"))
    return Pipeline(cfg, config.build(cfg))


def _error(status: int, reason: str, detail: str, pages: list[int] | None = None) -> JSONResponse:
    body: dict = {"status": status, "reason": reason, "detail": detail}
    if pages:
        body["pages"] = pages
    return JSONResponse(body, status_code=status)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    pipeline = get_pipeline()
    return {
        "service": "aiscr-atr",
        "version": __version__,
        "contract": {"name": "atr/docproc-json", "version": CONTRACT_VERSION},
        "components": [c.__dict__ for c in pipeline._components_info()],
    }


@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    forced_pages: str = Form("{}", description='JSON {"číslo strany": "důvod"}'),
    sha512: str | None = Form(None),
    paradata: bool = Form(False, description="vrátit {docproc, paradata} místo samotného výstupu"),
) -> Response:
    try:
        forced = {int(k): str(v) for k, v in json.loads(forced_pages).items()}
    except (ValueError, AttributeError) as exc:
        return _error(400, "invalid_forced_pages", str(exc))

    document = DocumentInput(
        data=await file.read(),
        filename=file.filename or "document",
        mime_type=file.content_type,
        sha512=sha512,
        forced_pages=forced,
    )
    try:
        output, para = get_pipeline().run_with_paradata(document)
        if paradata:
            # docproc jako kanonické bajty, aby seděl otisk SHA-512 v paradatech
            body = b'{"docproc":' + canonical_json(output) + b',"paradata":' + canonical_json(para) + b"}"
        else:
            body = canonical_json(output)
        return Response(body, media_type="application/json")
    except InputError as exc:
        return _error(422, exc.reason, exc.detail)
    except DependencyError as exc:
        return _error(503, "service_unavailable", str(exc))
    except PipelineError as exc:
        status = {"invalid_forced_pages": 400, "invalid_output": 500}.get(exc.reason, 422)
        return _error(status, exc.reason, exc.detail, exc.pages)
