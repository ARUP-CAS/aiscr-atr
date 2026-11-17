# Rozhraní kontejnerů s modely

Každý model běží ve vlastním kontejneru (`models/<jméno>/`). Hlavní služba
s ním mluví jen přes toto rozhraní, takže model jde vyměnit bez změny potrubí.

## Společné minimum

| Endpoint | Odpověď |
|---|---|
| `GET /health` | `{"status": "ok"}`, až je model načtený |
| `GET /info` | `{"model": "...", "version": "...", "licence": "...", "labels": [...]}` |
| `POST /predict` | podle typu služby níže |

## Rozvržení (`layout`)

Požadavek:

```json
{"image_png_base64": "..."}
```

Odpověď: bloky v pixelech obrázku, počátek vlevo nahoře. Převod na body dělá
adaptér `atr/adapters/layout/http.py`.

```json
{
  "blocks": [
    {"label": "text", "bbox_px": [120, 80, 900, 300], "confidence": 0.97, "order": 0, "polygon_px": null}
  ]
}
```

`order` je pořadí čtení, pokud ho model dává (PP-DocLayoutV3 ano), jinak `null`.
`polygon_px` je obrys bloku, pokud ho služba vrací (varianta s PyTorch ano, ONNX ne); potrubí ho nepoužívá.
Chyba 4xx znamená, že služba vstup odmítla (potrubí ji hlásí jako trvalé selhání, 422),
5xx a 429 dočasnou nedostupnost (503).
Třídy modelu se na typy bloků mapují v `config/pipeline.yaml` (`label_map`).

## Klasifikace stran (plán)

Služba ani adaptér zatím neexistují; port `page_classifier` je připravený. Zamýšlené rozhraní:
požadavek stejný jako u rozvržení, odpověď

```json
{"label": "text", "confidence": 0.93, "top": [{"label": "text", "confidence": 0.93}]}
```

## Rozpoznání bloků (`ocr-*`)

Modely za vLLM nebo SGLang s rozhraním kompatibilním s OpenAI
(`POST /v1/chat/completions`), stejně mluví i llama.cpp. Vlastní `/predict` nemají; adaptér
`atr/adapters/recognizer/openai.py` posílá výřez bloku jako obrázek a pokyn
úlohy (`OCR:`, `Table Recognition:`, …) podle konfigurace.
