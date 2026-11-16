"""Testes do agregador da ablação do BERTimbau."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.ablation_bert import collect_rows
from src.predictions import save_predictions

PER_CLASS = {"negativo": 0.8, "neutro": 0.4, "positivo": 0.9}


def write_subset(directory, n_train, macro_f1):
    directory.mkdir(parents=True, exist_ok=True)
    metrics = {"macro_f1": macro_f1, "accuracy": macro_f1 + 0.1, "f1_per_class": PER_CLASS}
    payload = {
        "config": {"n_train": n_train},
        "training": {"train_seconds": 12.5, "best_epoch": 2.0},
        "val": metrics,
        "test": metrics | {"macro_f1": macro_f1 + 0.01},
    }
    (directory / f"n{n_train}.json").write_text(json.dumps(payload), encoding="utf-8")


def write_predictions(path):
    frame = pd.DataFrame(
        {
            "split": ["val"] * 3 + ["test"] * 3,
            "row_id": range(6),
            "y_true": [0, 1, 2, 0, 1, 2],
            "y_pred": [0, 1, 2, 0, 1, 2],  # perfeito: macro-F1 = 1
        }
    )
    save_predictions(frame, path)


def test_curva_ordenada_com_pontos_dos_subconjuntos_e_o_cheio_das_previsoes(tmp_path):
    subsets = tmp_path / "ab"
    write_subset(subsets, 50472, 0.74)
    write_subset(subsets, 10094, 0.70)
    preds = tmp_path / "preds.csv"
    write_predictions(preds)

    rows = collect_rows(subsets, preds, n_train_full=100944)

    assert [r["n_train"] for r in rows] == [10094, 50472, 100944]
    assert [r["fraction"] for r in rows] == [0.1, 0.5, 1.0]
    assert rows[0]["test"]["macro_f1"] == pytest.approx(0.71)
    assert rows[-1]["test"]["macro_f1"] == 1.0
    assert rows[-1]["source"] == "previsões"
    assert set(rows[0]) >= {"fraction", "n_train", "val", "test", "best_epoch"}


def test_sem_previsoes_a_curva_so_tem_os_subconjuntos(tmp_path):
    write_subset(tmp_path / "ab", 25236, 0.72)
    rows = collect_rows(tmp_path / "ab", tmp_path / "nao_existe.csv", n_train_full=100944)
    assert [r["n_train"] for r in rows] == [25236]


def test_sem_nada_para_agregar_da_erro_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="nada para agregar"):
        collect_rows(tmp_path / "vazio", tmp_path / "nao_existe.csv", n_train_full=100944)
