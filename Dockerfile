FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY atr ./atr
RUN pip install --no-cache-dir .
COPY config ./config

RUN useradd --uid 1000 --create-home atr
USER atr
EXPOSE 8000
CMD ["uvicorn", "atr.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
