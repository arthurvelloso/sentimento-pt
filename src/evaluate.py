"""Avaliador único: todos os modelos do projeto passam por aqui.

TF-IDF, BERTimbau e LLM são comparados com exatamente o mesmo código, o mesmo
conjunto de teste e as mesmas métricas. Isso evita que a comparação mude de
resultado por diferença de cálculo.

Um modelo é qualquer função ``predict_fn(textos) -> classes`` (lista de textos
entra, array de inteiros sai). O resultado é um dicionário serializável em JSON.

A latência é medida por exemplo (um texto por chamada, como numa API), e não
pelo tempo do lote. Ela depende da máquina; as métricas de qualidade não.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

PredictFn = Callable[[list[str]], np.ndarray]


def measure_latency(
    predict_fn: PredictFn,
    texts: Sequence[str],
    *,
    n_samples: int = 300,
    warmup: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Latência por exemplo, em milissegundos (p50, p95 e média).

    Sorteia ``n_samples`` textos e chama ``predict_fn`` com um por vez. As
    primeiras ``warmup`` chamadas não entram na conta (cache, JIT, alocação).
    """
    if len(texts) == 0:
        raise ValueError("sem textos para medir latência")
    rng = np.random.default_rng(seed)
    size = min(n_samples + warmup, len(texts))
    picked = rng.choice(len(texts), size=size, replace=False)

    times: list[float] = []
    for position, index in enumerate(picked):
        start = time.perf_counter()
        predict_fn([texts[int(index)]])
        elapsed_ms = (time.perf_counter() - start) * 1000
        if position >= warmup or size <= warmup:
            times.append(elapsed_ms)
    arr = np.asarray(times)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "n_samples": int(arr.size),
    }


def evaluate(
    predict_fn: PredictFn,
    texts: Sequence[str],
    y_true: Sequence[int],
    labels: Sequence[str],
    *,
    name: str,
    latency_samples: int = 300,
) -> dict[str, Any]:
    """Avalia um modelo: macro-F1, acurácia, F1 por classe, matriz de confusão e latência."""
    texts = list(texts)
    y = np.asarray(y_true)
    if len(texts) != len(y):
        raise ValueError(f"textos ({len(texts)}) e rótulos ({len(y)}) com tamanhos diferentes")
    if len(texts) == 0:
        raise ValueError("conjunto de avaliação vazio")

    y_pred = np.asarray(predict_fn(texts))
    if y_pred.shape != y.shape:
        raise ValueError(f"predict_fn devolveu shape {y_pred.shape}, esperado {y.shape}")

    classes = list(range(len(labels)))
    per_class = f1_score(y, y_pred, labels=classes, average=None, zero_division=0)
    matrix = confusion_matrix(y, y_pred, labels=classes)
    support = np.bincount(y, minlength=len(labels))

    return {
        "name": name,
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, labels=classes, average="macro", zero_division=0)),
        "f1_per_class": {labels[i]: float(per_class[i]) for i in classes},
        "support": {labels[i]: int(support[i]) for i in classes},
        "confusion_matrix": {
            "labels": list(labels),
            "rows": "classe verdadeira",
            "cols": "classe prevista",
            "values": matrix.tolist(),
        },
        "latency": measure_latency(predict_fn, texts, n_samples=latency_samples),
    }


def save_result(result: dict[str, Any], path: Path | str) -> None:
    """Grava o resultado em JSON legível."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
