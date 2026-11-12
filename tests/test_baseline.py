"""Teste do baseline TF-IDF em um corpus sintético pequeno."""

from __future__ import annotations

import numpy as np

from src.baseline_tfidf import fit_and_select
from src.evaluate import evaluate

LABELS = ("negativo", "neutro", "positivo")
WORDS = {
    0: ["horrível", "péssimo", "quebrou", "atrasou", "decepção"],
    1: ["razoável", "mediano", "normal", "comum", "regular"],
    2: ["ótimo", "excelente", "adorei", "perfeito", "recomendo"],
}
FILLER = ["produto", "chegou", "entrega", "loja", "compra", "preço", "embalagem"]

CFG = {
    "word_ngram": [1, 2],
    "word_min_df": 1,
    "word_max_features": 5000,
    "char_ngram": [2, 4],
    "char_min_df": 1,
    "char_max_features": 5000,
    "max_iter": 200,
    "grid": {"C": [1.0, 10.0], "class_weight": [None, "balanced"]},
}


def corpus(n: int, seed: int):
    rng = np.random.default_rng(seed)
    labels = rng.choice([0, 1, 2], size=n, p=[0.3, 0.2, 0.5])
    texts = []
    for label in labels:
        picked = list(rng.choice(WORDS[int(label)], size=2)) + list(rng.choice(FILLER, size=3))
        rng.shuffle(picked)
        texts.append(" ".join(picked))
    return texts, labels


def test_baseline_aprende_o_sinal_e_registra_a_grade():
    x_train, y_train = corpus(600, seed=0)
    x_val, y_val = corpus(200, seed=1)

    pipeline, records = fit_and_select(x_train, y_train, x_val, y_val, CFG, seed=0)

    assert len(records) == 4  # 2 valores de C x 2 de class_weight
    assert {r["class_weight"] for r in records} == {None, "balanced"}
    assert all(0.0 <= r["val_macro_f1"] <= 1.0 for r in records)
    best = max(r["val_macro_f1"] for r in records)
    assert best > 0.95  # as palavras de cada classe são bem distintas

    result = evaluate(pipeline.predict, x_val, y_val, LABELS, name="sintetico", latency_samples=10)
    assert result["macro_f1"] == best
    assert pipeline.predict(["adorei, recomendo"]).shape == (1,)


def test_baseline_escolhe_a_melhor_combinacao_da_grade():
    x_train, y_train = corpus(600, seed=0)
    x_val, y_val = corpus(200, seed=1)
    pipeline, records = fit_and_select(x_train, y_train, x_val, y_val, CFG, seed=0)

    chosen = pipeline.named_steps["clf"]
    best = max(records, key=lambda r: r["val_macro_f1"])
    assert chosen.C == best["C"]
    assert chosen.class_weight == best["class_weight"]
