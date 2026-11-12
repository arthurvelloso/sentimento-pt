"""Testes do avaliador único."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.evaluate import evaluate, measure_latency, save_result

LABELS = ("negativo", "neutro", "positivo")


def perfect(y):
    y = np.asarray(y)
    lookup = {f"t{i}": int(v) for i, v in enumerate(y)}
    return lambda texts: np.array([lookup[t] for t in texts])


def test_modelo_perfeito():
    y = [0, 1, 2, 2, 1, 0]
    texts = [f"t{i}" for i in range(len(y))]
    result = evaluate(perfect(y), texts, y, LABELS, name="perfeito", latency_samples=3)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert all(v == 1.0 for v in result["f1_per_class"].values())
    assert result["confusion_matrix"]["values"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_valores_conhecidos_de_matriz_e_f1():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    texts = [f"t{i}" for i in range(6)]
    result = evaluate(lambda ts: y_pred, texts, y_true, LABELS, name="x", latency_samples=2)

    assert result["confusion_matrix"]["values"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert result["accuracy"] == pytest.approx(4 / 6)
    # F1 por classe: negativo 0.5, neutro 0.8, positivo 0.6667
    assert result["f1_per_class"]["negativo"] == pytest.approx(0.5)
    assert result["f1_per_class"]["neutro"] == pytest.approx(0.8)
    assert result["f1_per_class"]["positivo"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert result["support"] == {"negativo": 2, "neutro": 2, "positivo": 2}


def test_classe_nunca_prevista_tem_f1_zero_sem_estourar():
    y_true = [0, 1, 2, 2]
    texts = [f"t{i}" for i in range(4)]
    result = evaluate(lambda ts: np.array([2, 2, 2, 2]), texts, y_true, LABELS, name="x")
    assert result["f1_per_class"]["negativo"] == 0.0
    assert result["f1_per_class"]["neutro"] == 0.0


def test_latencia_tem_campos_e_ordem_coerente():
    texts = [f"t{i}" for i in range(50)]
    latency = measure_latency(lambda ts: np.zeros(len(ts), dtype=int), texts, n_samples=20)
    assert set(latency) == {"p50_ms", "p95_ms", "mean_ms", "n_samples"}
    assert latency["n_samples"] == 20
    assert 0 <= latency["p50_ms"] <= latency["p95_ms"]


def test_latencia_chama_o_modelo_um_texto_por_vez():
    seen: list[int] = []

    def predict(texts):
        seen.append(len(texts))
        return np.zeros(len(texts), dtype=int)

    measure_latency(predict, ["a", "b", "c", "d"], n_samples=3, warmup=1)
    assert set(seen) == {1}


def test_erros_claros_para_entradas_invalidas():
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        evaluate(lambda ts: np.zeros(2), ["a", "b"], [0, 1, 2], LABELS, name="x")
    with pytest.raises(ValueError, match="vazio"):
        evaluate(lambda ts: np.zeros(0), [], [], LABELS, name="x")
    with pytest.raises(ValueError, match="shape"):
        evaluate(lambda ts: np.zeros(5), ["a", "b"], [0, 1], LABELS, name="x")


def test_save_result_grava_json_legivel(tmp_path):
    y = [0, 1, 2]
    result = evaluate(perfect(y), ["t0", "t1", "t2"], y, LABELS, name="x", latency_samples=2)
    path = tmp_path / "sub" / "r.json"
    save_result(result, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["macro_f1"] == 1.0
    assert loaded["confusion_matrix"]["labels"] == list(LABELS)
