"""HTTP rozhraní služby: /convert, /info, /health.

Chybový model jako ostatní služby ATRIUM: {"status", "reason", "detail"}.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from atr import __version__, config, registry
from atr.adapters.assembler.docproc import CONTRACT_VERSION
from atr.paradata import canonical_json
from atr.pipeline import Pipeline, PipelineError
from atr.ports.errors import DependencyError, InputError
from atr.ports.types import DocumentInput


@asynccontextmanager
async def _lifespan(_: FastAPI):
    # Konfiguraci a modely v procesu (jazyk) načíst při startu, ne při prvním požadavku:
    # chybná konfigurace službu nespustí a první /convert neblokuje /health.
    await run_in_threadpool(get_pipeline)
    yield


app = FastAPI(
    title="aiscr-atr",
    lifespan=_lifespan,
    version=__version__,
    description=(
        "Otevřená služba ATR: z dokumentu (PDF, sken, obrázek) vytvoří `atr/docproc-json` "
        "(rozvržení, bloky s typem, polohou, pořadím čtení a textem; routing každé strany). "
        "Chyby mají tvar `{status, reason, detail}`: 400 neplatný požadavek, 413 příliš velký soubor, "
        "422 nečitelný vstup nebo trvalé selhání zpracování (`all_engines_failed`, `layout_rejected`; "
        "opakování nepomůže), 503 dočasně nedostupná služba modelu (zopakovat), 500 neplatný výstup."
    ),
    license_info={"name": "MIT"},
)


MAX_UPLOAD_MB = int(os.environ.get("ATR_MAX_UPLOAD_MB", "1024"))
MAX_CONCURRENT = int(os.environ.get("ATR_MAX_CONCURRENT", "2"))
_semaphore: asyncio.Semaphore | None = None


def _slots() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:  # vytvořit až v běžící smyčce událostí
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    return _semaphore


async def _read_limited(file: UploadFile) -> bytes | None:
    """Přečte nahraný soubor po částech; None, když překročí ATR_MAX_UPLOAD_MB."""
    limit = MAX_UPLOAD_MB * 1024 * 1024
    chunks, size = [], 0
    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


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

    try:
        # Potrubí je synchronní (PDFium, blokující HTTP): běží ve vlákně, aby služba
        # dál odpovídala na /health a /info. Souběžné převody omezuje semafor; soubor
        # se do paměti čte až s volným místem (do té doby leží v dočasném souboru).
        async with _slots():
            data = await _read_limited(file)
            if data is None:
                return _error(413, "payload_too_large", f"soubor je větší než {MAX_UPLOAD_MB} MB")
            document = DocumentInput(
                data=data,
                filename=file.filename or "document",
                mime_type=file.content_type,
                sha512=sha512,
                forced_pages=forced,
            )
            del data
            output, para = await run_in_threadpool(get_pipeline().run_with_paradata, document)
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
