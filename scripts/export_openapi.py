"""Uloží specifikaci OpenAPI služby do docs/openapi.json (akceptační kritérium aiscr-atr#2).

  python scripts/export_openapi.py          # zapíše
  python scripts/export_openapi.py --check  # ověří, že soubor odpovídá kódu (CI)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from atr.api.app import app

OUT = Path(__file__).resolve().parent.parent / "docs" / "openapi.json"


def main() -> int:
    spec = json.dumps(app.openapi(), ensure_ascii=False, indent=1) + "\n"
    if "--check" in sys.argv:
        if not OUT.is_file() or OUT.read_text(encoding="utf-8") != spec:
            print("docs/openapi.json neodpovídá kódu; spusť python scripts/export_openapi.py", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(spec, encoding="utf-8")
    print(f"zapsáno {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
