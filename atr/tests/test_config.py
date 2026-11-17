from pathlib import Path

import pytest

from atr import config, registry

registry.load_builtin_adapters()

ROOT = Path(__file__).resolve().parents[2]


def test_repository_config_is_valid_and_adapters_exist():
    cfg = config.load(ROOT / "config" / "pipeline.yaml")
    for port, spec in cfg.adapters.items():
        assert spec.adapter in registry.available(port), f"{port}/{spec.adapter}"
    for name, spec in cfg.recognizers.items():
        assert spec.adapter in registry.available("recognizer"), name


def test_missing_required_port_is_reported():
    with pytest.raises(config.ConfigError, match="layout"):
        config.parse({"ports": {p: {"adapter": "fake"} for p in config.REQUIRED_PORTS if p != "layout"}})


def test_unknown_block_type_is_reported():
    raw = {"ports": {p: {"adapter": "fake"} for p in config.REQUIRED_PORTS},
           "recognizers": {"x": {"adapter": "fake"}}, "recognize": {"paragraph": "x"}}
    with pytest.raises(config.ConfigError, match="paragraph"):
        config.parse(raw)


def test_unknown_adapter_lists_available():
    with pytest.raises(registry.RegistryError, match="dostupné"):
        registry.create("layout", "neexistuje")
