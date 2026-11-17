# Kontejnery s modely

Jeden model = jedna složka = jeden obraz. Rozhraní je v
[`docs/model-services.md`](../docs/model-services.md).

| Složka | Model | Licence | Stav |
|---|---|---|---|
| `layout-ppdoclayoutv3/` | PP-DocLayoutV3 (`PaddlePaddle/PP-DocLayoutV3_safetensors`, připnutá revize); `Dockerfile.onnx` = ONNX Runtime na CPU (výchozí, 773 MB), `Dockerfile` = PyTorch pro GPU (2,3 GB) | Apache-2.0 | hotovo |
| služba `ocr-text` v compose | PaddleOCR-VL-1.6 za vLLM (připnutá revize) nebo llama.cpp (profil `vulkan`) | Apache-2.0 | běží; výběr modelu čeká na vyhodnocení |
| služba `ocr-alt` v compose | GLM-OCR za vLLM (připnutá revize) nebo llama.cpp (profil `vulkan-alt`) | MIT | běží; výběr modelu čeká na vyhodnocení |
| `pageclass-vit/` (plán) | `ufal/vit-historical-page` (ATRIUM, ÚFAL) | MIT | fáze 7; složka, služba v compose ani adaptér zatím neexistují |

Nový model: přidej složku s `Dockerfile` a tenkým serverem podle rozhraní,
službu do `docker-compose.yml` a adaptér a řádek do `config/pipeline.yaml`.
