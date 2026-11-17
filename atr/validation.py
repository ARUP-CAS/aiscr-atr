"""Validace výstupu proti schématu `atr/docproc-json` a kontrola úplnosti.

Schéma je zatím návrh v `atr/schema/docproc.schema.json` (normativní verze vznikne
v aiscr-docs-pipeline#17). Kromě schématu se kontroluje, co JSON Schema neumí:
každá strana originálu je v routing právě jednou, strany v `pages` odpovídají
neprázdným stranám a pořadí čtení je v každé straně 0..n-1.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import jsonschema

SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "docproc.schema.json"


class OutputInvalid(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors[:5]))
        self.errors = errors


@lru_cache(maxsize=1)
def _validator() -> jsonschema.protocols.Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def errors(output: dict) -> list[str]:
    found = [f"{'/'.join(map(str, e.absolute_path)) or '(kořen)'}: {e.message}"
             for e in sorted(_validator().iter_errors(output), key=lambda e: list(e.absolute_path))]
    if found:
        return found
    routing = output["metadata"]["routing"]["pages"]
    numbers = [r["page"] for r in routing]
    if numbers != list(range(1, output["pageCount"] + 1)):
        found.append(f"routing neobsahuje každou stranu 1..{output['pageCount']} právě jednou: {numbers}")
    with_content = [r["page"] for r in routing if r["route"] != "empty"]
    pages = [p["pageNumber"] for p in output["pages"]]
    if pages != with_content:
        found.append(f"pages {pages} neodpovídají neprázdným stranám v routing {with_content}")
    for page in output["pages"]:
        order = sorted(b["readingOrder"] for b in page["blocks"])
        if order != list(range(len(order))):
            found.append(f"strana {page['pageNumber']}: pořadí čtení není 0..{len(order) - 1}")
        for b in page["blocks"]:
            x0, y0, x1, y1 = b["bbox"]
            if x1 < x0 or y1 < y0 or x1 > page["width"] + 0.5 or y1 > page["height"] + 0.5:
                found.append(f"strana {page['pageNumber']}: blok mimo stranu {b['bbox']}")
                break
    return found


def validate(output: dict) -> None:
    problems = errors(output)
    if problems:
        raise OutputInvalid(problems)
