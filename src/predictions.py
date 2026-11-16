"""Previsões por exemplo, salvas sem o texto.

Cada linha é ``(split, row_id, y_true, y_pred)`` mais, se houver, as probabilidades
por classe. O ``row_id`` é a linha no CSV original da B2W; o texto fica fora do
repositório (a base é CC BY-NC-SA 4.0), então estes arquivos podem ser versionados e
servem para a análise de erros sem republicar nenhuma avaliação.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("split", "row_id", "y_true", "y_pred")


def make_frame(
    split: str,
    row_ids: Sequence[int],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    proba: np.ndarray | None = None,
    labels: Sequence[str] = (),
) -> pd.DataFrame:
    """Monta o DataFrame de previsões de um split.

    ``y_pred = -1`` marca resposta inválida (usado pelo LLM quando a saída não é um rótulo).
    """
    if not (len(row_ids) == len(y_true) == len(y_pred)):
        raise ValueError("row_ids, y_true e y_pred precisam ter o mesmo tamanho")
    frame = pd.DataFrame(
        {
            "split": split,
            "row_id": np.asarray(row_ids, dtype=int),
            "y_true": np.asarray(y_true, dtype=int),
            "y_pred": np.asarray(y_pred, dtype=int),
        }
    )
    if proba is not None:
        proba = np.asarray(proba)
        if proba.shape != (len(frame), len(labels)):
            raise ValueError(f"proba com shape {proba.shape}, esperado {(len(frame), len(labels))}")
        for index, label in enumerate(labels):
            frame[f"p_{label}"] = proba[:, index].round(6)
    return frame


def save_predictions(frame: pd.DataFrame, path: Path | str) -> None:
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"faltam colunas: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_predictions(path: Path | str, split: str | None = None) -> pd.DataFrame:
    """Lê um CSV de previsões, opcionalmente só de um split."""
    frame = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: faltam colunas {missing}")
    if split is not None:
        frame = frame[frame["split"] == split].reset_index(drop=True)
    return frame
