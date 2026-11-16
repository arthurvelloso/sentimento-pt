"""Testes da ablação do TF-IDF e do módulo de previsões, com corpus sintético."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ablation_tfidf import run_ablation
from src.predictions import load_predictions, make_frame, save_predictions

WORDS = {
    0: ["horrivel", "pessimo", "quebrou", "atrasou", "decepcao"],
    1: ["razoavel", "mediano", "normal", "comum", "regular"],
    2: ["otimo", "excelente", "adorei", "perfeito", "recomendo"],
}
FILLER = ["produto", "chegou", "entrega", "loja", "compra", "preco", "embalagem"]
CFG = {
    "word_ngram": [1, 2],
    "word_min_df": 1,
    "word_max_features": 5000,
    "char_ngram": [2, 4],
    "char_min_df": 1,
    "char_max_features": 5000,
    "max_iter": 200,
}


def corpus(n: int, seed: int, start: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.choice([0, 1, 2], size=n, p=[0.3, 0.2, 0.5])
    texts = []
    for label in labels:
        picked = list(rng.choice(WORDS[int(label)], size=2)) + list(rng.choice(FILLER, size=3))
        rng.shuffle(picked)
        texts.append(" ".join(picked))
    return pd.DataFrame({"row_id": np.arange(start, start + n), "text": texts, "label": labels})


BEST = {"C": 1.0, "class_weight": "balanced"}


def test_ablacao_treina_com_a_fracao_certa_e_devolve_previsoes_do_ponto_cheio():
    train, val, test = corpus(600, 0), corpus(150, 1, 1000), corpus(150, 2, 2000)
    rows, predictions = run_ablation(train, val, test, CFG, BEST, seed=0, fractions=(0.25, 1.0))

    assert [r["n_train"] for r in rows] == [150, 600]
    assert all(r["val"]["macro_f1"] > 0.8 for r in rows)  # sinal forte mesmo com pouco dado
    assert rows[1]["converged"] is True

    assert predictions is not None
    assert set(predictions["split"]) == {"val", "test"}
    assert len(predictions) == 300
    assert set(predictions.columns) >= {"split", "row_id", "y_true", "y_pred"}
    assert set(predictions["row_id"]) == set(val["row_id"]) | set(test["row_id"])


def test_sem_ponto_de_100_por_cento_nao_ha_previsoes():
    train, val, test = corpus(300, 0), corpus(60, 1, 1000), corpus(60, 2, 2000)
    _, predictions = run_ablation(train, val, test, CFG, BEST, seed=0, fractions=(0.5,))
    assert predictions is None


def test_ablacao_e_reprodutivel():
    train, val, test = corpus(300, 0), corpus(60, 1, 1000), corpus(60, 2, 2000)
    first, preds_a = run_ablation(train, val, test, CFG, BEST, seed=3, fractions=(0.5, 1.0))
    second, preds_b = run_ablation(train, val, test, CFG, BEST, seed=3, fractions=(0.5, 1.0))
    assert [r["test"]["macro_f1"] for r in first] == [r["test"]["macro_f1"] for r in second]
    assert preds_a["y_pred"].tolist() == preds_b["y_pred"].tolist()


# ------------------------------------------------------------------ previsões


def test_previsoes_ida_e_volta_e_filtro_por_split(tmp_path):
    val = make_frame("val", [1, 2], [0, 1], [0, 2])
    test = make_frame(
        "test",
        [10, 11],
        [2, 2],
        [2, -1],
        proba=np.array([[0.1, 0.1, 0.8], [0.3, 0.3, 0.4]]),
        labels=("negativo", "neutro", "positivo"),
    )
    path = tmp_path / "p" / "preds.csv"
    save_predictions(pd.concat([val, test], ignore_index=True), path)

    loaded = load_predictions(path)
    assert len(loaded) == 4
    only_test = load_predictions(path, split="test")
    assert only_test["row_id"].tolist() == [10, 11]
    assert only_test["y_pred"].tolist() == [2, -1]  # -1 = resposta inválida
    assert only_test["p_positivo"].tolist() == [0.8, 0.4]


def test_previsoes_validam_tamanhos_e_colunas(tmp_path):
    with pytest.raises(ValueError, match="mesmo tamanho"):
        make_frame("val", [1, 2], [0], [0, 1])
    with pytest.raises(ValueError, match="proba"):
        make_frame("val", [1], [0], [0], proba=np.zeros((1, 2)), labels=("a", "b", "c"))
    with pytest.raises(ValueError, match="faltam colunas"):
        save_predictions(pd.DataFrame({"a": [1]}), tmp_path / "x.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="faltam colunas"):
        load_predictions(bad)
