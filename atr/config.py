"""Načtení konfigurace potrubí a sestavení adaptérů podle ní."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from atr import registry
from atr.ports.types import BlockType

# Porty, bez kterých potrubí neběží. Ostatní jsou nepovinné.
REQUIRED_PORTS = (
    "page_source",
    "renderer",
    "text_layer",
    "text_layer_judge",
    "router",
    "layout",
    "text_assigner",
    "assembler",
)


@dataclass(frozen=True)
class AdapterSpec:
    adapter: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    adapters: dict[str, AdapterSpec]
    # typ bloku → jméno rozpoznávače (klíč v `recognizers`); typ bez záznamu se nerozpoznává
    recognize: dict[BlockType, str]
    recognizers: dict[str, AdapterSpec]
    render_dpi: float = 200.0
    crop_padding_pt: float = 2.0


class ConfigError(ValueError):
    pass


def _spec(raw: Any, where: str) -> AdapterSpec:
    if not isinstance(raw, dict) or "adapter" not in raw:
        raise ConfigError(f"{where}: čekám slovník s klíčem 'adapter'")
    return AdapterSpec(adapter=str(raw["adapter"]), params=dict(raw.get("params") or {}))


def parse(raw: dict[str, Any]) -> PipelineConfig:
    ports = raw.get("ports") or {}
    unknown = set(ports) - set(registry.PORTS) - {"recognizer"}
    if unknown:
        raise ConfigError(f"neznámé porty v konfiguraci: {', '.join(sorted(unknown))}")
    missing = [p for p in REQUIRED_PORTS if p not in ports]
    if missing:
        raise ConfigError(f"chybí povinné porty: {', '.join(missing)}")
    adapters = {port: _spec(spec, f"ports.{port}") for port, spec in ports.items()}

    recognizers = {
        name: _spec(spec, f"recognizers.{name}") for name, spec in (raw.get("recognizers") or {}).items()
    }
    recognize: dict[BlockType, str] = {}
    for type_name, recognizer in (raw.get("recognize") or {}).items():
        try:
            block_type = BlockType(type_name)
        except ValueError:
            raise ConfigError(f"recognize: neznámý typ bloku {type_name!r}") from None
        if recognizer not in recognizers:
            raise ConfigError(f"recognize.{type_name}: rozpoznávač {recognizer!r} není v 'recognizers'")
        recognize[block_type] = recognizer

    pipeline = raw.get("pipeline") or {}
    return PipelineConfig(
        adapters=adapters,
        recognize=recognize,
        recognizers=recognizers,
        render_dpi=float(pipeline.get("render_dpi", 200)),
        crop_padding_pt=float(pipeline.get("crop_padding_pt", 2)),
    )


def load(path: str | Path) -> PipelineConfig:
    with open(path, encoding="utf-8") as fh:
        return parse(yaml.safe_load(fh) or {})


@dataclass
class Components:
    """Sestavené adaptéry podle konfigurace."""

    adapters: dict[str, Any]
    recognizers: dict[str, Any]

    def get(self, port: str) -> Any | None:
        return self.adapters.get(port)

    def __getitem__(self, port: str) -> Any:
        return self.adapters[port]


def build(config: PipelineConfig) -> Components:
    adapters = {port: registry.create(port, s.adapter, s.params) for port, s in config.adapters.items()}
    recognizers = {
        name: registry.create("recognizer", s.adapter, s.params) for name, s in config.recognizers.items()
    }
    return Components(adapters=adapters, recognizers=recognizers)
