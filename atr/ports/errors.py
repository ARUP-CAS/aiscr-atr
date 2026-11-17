"""Chyby, které adaptéry hlásí potrubí a API."""

from __future__ import annotations


class InputError(ValueError):
    """Vstup nejde zpracovat (neznámý formát, poškozený nebo zašifrovaný soubor)."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class DependencyError(RuntimeError):
    """Služba modelu (rozvržení, rozpoznávání) není dostupná. Úlohu jde zopakovat."""

    def __init__(self, service: str, detail: str) -> None:
        super().__init__(f"{service}: {detail}")
        self.service = service
        self.detail = detail


class ProcessingError(RuntimeError):
    """Trvalé selhání zpracování (model vstup odmítl, výstup je neúplný). Opakování nepomůže."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def classify_http_error(service: str, url: str, exc: Exception) -> Exception:
    """httpx chyba → DependencyError (dočasná: spojení, timeout, 5xx, 429) nebo
    ProcessingError (trvalá: ostatní 4xx, služba vstup odmítla)."""
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code < 500 and code not in (408, 429):
            body = exc.response.text[:300]
            return ProcessingError(f"{service}_rejected", f"HTTP {code} z {url}: {body}")
    return DependencyError(service, f"{url}: {exc}")
