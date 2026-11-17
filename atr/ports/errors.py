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
