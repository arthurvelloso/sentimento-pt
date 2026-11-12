"""Leitura da configuração do projeto."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Carrega e valida o YAML de configuração."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config não encontrada: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("a config precisa ser um mapeamento (chave: valor)")

    for key in ("seed", "data", "split", "tfidf"):
        if key not in config:
            raise ValueError(f"campo obrigatório ausente na config: {key}")

    fractions = [config["split"][name] for name in ("train", "val", "test")]
    if abs(sum(fractions) - 1.0) > 1e-9 or min(fractions) <= 0:
        raise ValueError(f"as frações do split precisam ser positivas e somar 1: {fractions}")
    return config
