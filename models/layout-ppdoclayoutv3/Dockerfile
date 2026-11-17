# PP-DocLayoutV3 přes transformers. Výchozí sestavení je pro CPU; pro GPU:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 .
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG MODEL_ID=PaddlePaddle/PP-DocLayoutV3_safetensors
ARG MODEL_REVISION=e4f489dc536556fc1be02b973ba06756e0d4a2ac

ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    MODEL_ID=${MODEL_ID} \
    MODEL_REVISION=${MODEL_REVISION}

RUN pip install --index-url ${TORCH_INDEX} torch==2.9.1 torchvision==0.24.1
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Váhy jsou součástí obrazu (připnutá revize), za běhu se nic nestahuje.
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_ID}', revision='${MODEL_REVISION}')"
ENV HF_HUB_OFFLINE=1

WORKDIR /app
COPY server.py .
RUN useradd --uid 1000 --create-home layout && chmod -R a+rX /opt/hf
USER layout
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s CMD python -c \
  "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
