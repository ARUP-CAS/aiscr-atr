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

## ArchaeoDocLayout (rozvržení se třídou `handwriting`)

Dotrénovaný PP-DocLayoutV3 je na Hugging Face jako
[`aiscr/ArchaeoDocLayout`](https://huggingface.co/aiscr/ArchaeoDocLayout) (Apache-2.0, značka `v0.1`;
postup a výsledky v [`training/layout/README.md`](../training/layout/README.md)).
Repozitář je zatím soukromý, proto výchozí compose používá původní PP-DocLayoutV3,
který třídu `handwriting` nezná. Obraz s ArchaeoDocLayout:

```bash
cd models/layout-ppdoclayoutv3
HF_TOKEN=… docker build --secret id=hf_token,env=HF_TOKEN -f Dockerfile.onnx \
  --build-arg MODEL_ID=aiscr/ArchaeoDocLayout --build-arg MODEL_REVISION=v0.1 \
  -t ghcr.io/arup-cas/aiscr-atr-layout-ppdoclayoutv3:archaeodoclayout-v0.1 .
```

Po zveřejnění repozitáře token odpadne a stačí změnit výchozí `MODEL_ID` a `MODEL_REVISION`.
