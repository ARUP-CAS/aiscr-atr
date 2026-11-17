#!/usr/bin/env bash
# Stáhne modely GGUF pro llama.cpp (profil vulkan) s připnutou revizí.
set -euo pipefail
DIR="${GGUF_DIR:-$HOME/.cache/aiscr-atr/gguf}"
get() {  # repo revize soubor cíl
  mkdir -p "$(dirname "$DIR/$4")"
  [ -s "$DIR/$4" ] && { echo "už je: $4"; return; }
  curl -fL --retry 3 -o "$DIR/$4" "https://huggingface.co/$1/resolve/$2/$3"
}
# PaddleOCR-VL-1.6 (Apache-2.0)
get PaddlePaddle/PaddleOCR-VL-1.6-GGUF 511b09642bb324401f15f97cc23bc67e8f0a291d PaddleOCR-VL-1.6-GGUF.gguf paddleocr-vl-1.6/model.gguf
get PaddlePaddle/PaddleOCR-VL-1.6-GGUF 511b09642bb324401f15f97cc23bc67e8f0a291d PaddleOCR-VL-1.6-GGUF-mmproj.gguf paddleocr-vl-1.6/mmproj.gguf
# GLM-OCR (MIT), převod Q8_0
get ggml-org/GLM-OCR-GGUF 65a42de1148dbed2297e922b5dbc7d9b70c36578 GLM-OCR-Q8_0.gguf glm-ocr/model.gguf
get ggml-org/GLM-OCR-GGUF 65a42de1148dbed2297e922b5dbc7d9b70c36578 mmproj-GLM-OCR-Q8_0.gguf glm-ocr/mmproj.gguf
echo "hotovo: $DIR"
