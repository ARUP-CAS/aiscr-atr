# Kontejnery s modely

Jeden model = jedna složka = jeden obraz. Rozhraní je v
[`docs/model-services.md`](../docs/model-services.md).

| Složka | Model | Licence | Stav |
|---|---|---|---|
| `layout-ppdoclayoutv3/` | PP-DocLayoutV3 (`PaddlePaddle/PP-DocLayoutV3_safetensors`, připnutá revize) přes transformers | Apache-2.0 | hotovo; CPU i GPU |
| `ocr-paddleocr-vl/` | PaddleOCR-VL-1.6 za vLLM | Apache-2.0 | fáze 4; zatím jen služba v `docker-compose.yml` |
| `ocr-glm-ocr/` | GLM-OCR za vLLM | MIT | fáze 4; zatím jen služba v `docker-compose.yml` |
| `pageclass-vit/` | `ufal/vit-historical-page` (ATRIUM, ÚFAL) | MIT | fáze 7 |

Nový model: přidej složku s `Dockerfile` a tenkým serverem podle rozhraní,
službu do `docker-compose.yml` a adaptér a řádek do `config/pipeline.yaml`.
