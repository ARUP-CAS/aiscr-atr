# aiscr-atr

Otevřená služba ATR pro projekt ATRIUM: z jakéhokoli dokumentu (born-digital PDF,
sken, smíšené PDF, obrázek) vytvoří `atr/docproc-json` podle kontraktu
[aiscr-docs-pipeline#17](https://github.com/ARUP-CAS/aiscr-docs-pipeline/issues/17).
Zadání je v [aiscr-atr#2](https://github.com/ARUP-CAS/aiscr-atr/issues/2), plán stavby
v [`docs/PLAN.md`](docs/PLAN.md).

## Jak je to poskládané

Potrubí (`atr/pipeline/`) zná jen **porty** (`atr/ports/`): pevná rozhraní pro čtení
stran, textovou vrstvu, směrování, rozvržení, rozpoznání bloků a další. Konkrétní
nástroje jsou **adaptéry** (`atr/adapters/<port>/`), vybrané v
[`config/pipeline.yaml`](config/pipeline.yaml). Modely běží každý ve vlastním
kontejneru (`models/`) s rozhraním podle [`docs/model-services.md`](docs/model-services.md).

Výměna modelu nebo knihovny = nový adaptér nebo kontejner + změna konfigurace.
Tok potrubí se nemění.

## Vývoj

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest
```

Testy používají adaptéry `fake` a nepotřebují PDF ani GPU.

## Spuštění

```bash
docker compose up atr-api                  # služba na http://localhost:8080
docker compose --profile gpu up            # + rozvržení a PaddleOCR-VL (potřebuje GPU a VLLM_IMAGE v .env)
```

Rozhraní: `POST /convert` (soubor, nepovinně `forced_pages` jako JSON
`{"číslo strany": "důvod"}` a `sha512`), `GET /info`, `GET /health`.
Chyby mají tvar `{"status", "reason", "detail"}`.

## Licence

Kód je pod licencí MIT. Modely mají vlastní licence, viz [`models/README.md`](models/README.md);
používají se jen modely s otevřenou licencí.
