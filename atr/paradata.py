"""Paradata k výstupu `atr/docproc-json`: čím, z čeho a jak výstup vznikl.

Tvar vychází z RO-Crate Process Run Crate (https://w3id.org/ro/wfrun/process):
jedna `CreateAction` s nástroji (`instrument`, každá komponenta potrubí zvlášť
s verzí, licencí a parametry), vstupem (`object`) a výsledkem (`result`).
Data o běhu, která do výstupu nepatří (kontrakt #17, bod 3), jsou tady.

Otisk výsledku je SHA-512 kanonické serializace výstupu (`canonical_json`):
UTF-8, bez mezer, pořadí klíčů zachované. Kdo výstup ukládá, ukládá právě tyto bajty.
"""

from __future__ import annotations

import hashlib
import json

from atr import __version__
from atr.ports.types import DocumentResult, Route

CONTEXT = [
    "https://w3id.org/ro/crate/1.1/context",
    {"wfrun": "https://w3id.org/ro/terms/workflow-run#"},
]


def canonical_json(output: dict) -> bytes:
    return json.dumps(output, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def build(result: DocumentResult, output: dict) -> dict:
    doc = result.document
    data = canonical_json(output)
    routes: dict[str, int] = {}
    for page in result.pages:
        routes[page.decision.route.value] = routes.get(page.decision.route.value, 0) + 1

    instruments = []
    for n, c in enumerate(result.components):
        entry = {
            # pořadové číslo drží @id jedinečné i pro více rozpoznávačů nad stejným modelem
            "@id": f"#component-{n:02d}-{c.port}",
            "@type": "SoftwareApplication",
            "name": c.name,
            "softwareVersion": c.version,
            "additionalType": c.port,
        }
        if c.licence:
            entry["license"] = c.licence
        if c.params:
            entry["wfrun:parameters"] = c.params
        instruments.append(entry)

    source = {"@id": "#source", "@type": "File", "name": doc.filename}
    if doc.sha512:
        source["sha512"] = doc.sha512
    if doc.mime_type:
        source["encodingFormat"] = doc.mime_type

    action = {
        "@id": "#atr-run",
        "@type": "CreateAction",
        "name": "ATR: rozvržení a text dokumentu",
        "agent": {"@id": "#atr-service", "@type": "SoftwareApplication",
                  "name": "aiscr-atr", "softwareVersion": __version__},
        "instrument": instruments,
        "object": [source],
        "result": [{
            "@id": "#result",
            "@type": "File",
            "additionalType": "atr/docproc-json",
            "encodingFormat": "application/json",
            "contentSize": len(data),
            "sha512": hashlib.sha512(data).hexdigest(),
            "conformsTo": f"atr/docproc-json {output.get('version')}",
        }],
        "actionStatus": "CompletedActionStatus",
        "startTime": result.started_at,
        "endTime": result.finished_at,
        "wfrun:summary": {
            "pages": len(result.pages),
            "routes": routes,
            "forced_pages": {str(k): v for k, v in sorted(doc.forced_pages.items())},
            "recognized_blocks": sum(len(p.blocks) for p in result.pages
                                     if p.decision.route in (Route.OCR, Route.FORCED)),
        },
    }
    return {"@context": CONTEXT, "@graph": [action]}
