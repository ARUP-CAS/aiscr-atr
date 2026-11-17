"""Registr adaptérů: jméno v konfiguraci → třída adaptéru.

Adaptér se registruje dekorátorem::

    @register("layout", "http")
    class HttpLayoutDetector: ...

a konfigurace ho vybere::

    layout:
      adapter: http
      params: {url: http://layout:8000}
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any

PORTS = (
    "page_source",
    "renderer",
    "text_layer",
    "text_layer_judge",
    "page_classifier",
    "router",
    "layout",
    "text_assigner",
    "recognizer",
    "language",
    "assembler",
)

_REGISTRY: dict[str, dict[str, type]] = {port: {} for port in PORTS}


class RegistryError(LookupError):
    pass


def register(port: str, name: str) -> Callable[[type], type]:
    if port not in _REGISTRY:
        raise RegistryError(f"neznámý port {port!r}, známé: {', '.join(PORTS)}")

    def decorator(cls: type) -> type:
        existing = _REGISTRY[port].get(name)
        if existing is not None and existing is not cls:
            raise RegistryError(f"adaptér {port}/{name} je už registrovaný ({existing.__qualname__})")
        _REGISTRY[port][name] = cls
        return cls

    return decorator


def available(port: str) -> list[str]:
    return sorted(_REGISTRY[port])


def create(port: str, name: str, params: dict[str, Any] | None = None) -> Any:
    try:
        cls = _REGISTRY[port][name]
    except KeyError:
        raise RegistryError(
            f"adaptér {port}/{name} není registrovaný; dostupné: {', '.join(available(port)) or 'žádné'}"
        ) from None
    return cls(**(params or {}))


def load_builtin_adapters() -> None:
    """Naimportuje všechny moduly v `atr.adapters`, aby se zaregistrovaly."""
    import atr.adapters as package

    for module in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
        importlib.import_module(module.name)
