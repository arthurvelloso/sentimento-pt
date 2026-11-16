"""Ablação do tamanho dos dados para o TF-IDF: 10%, 25%, 50% e 100% do treino.

Usa a combinação (C, class_weight) escolhida no baseline, sem reabrir a grade, e só
muda quantos exemplos o modelo vê. O vetorizador é reajustado em cada subconjunto.

O ponto de 100% é o próprio baseline: serve de conferência (precisa reproduzir o
macro-F1 já registrado em ``results/tfidf.json``) e fornece as previsões por exemplo
usadas na análise de erros (``results/preds_tfidf.csv``).

A curva lê o teste uma vez por tamanho, para descrever a tendência; nenhuma escolha
é feita com ele.

Rode com ``make ablation-tfidf``.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.baseline_tfidf import build_features
from src.config import load_config
from src.data import LABELS, load_splits, stratified_subset
from src.evaluate import evaluate
from src.predictions import make_frame, save_predictions

FRACTIONS: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00)
RESULTS_PATH = Path("results/ablation_tfidf.json")
PREDICTIONS_PATH = Path("results/preds_tfidf.csv")


def _cached(pipeline: Pipeline, full_pred: np.ndarray) -> Any:
    """Predição que reaproveita o lote completo já calculado e delega o resto (latência)."""
    size = len(full_pred)

    def predict(texts: list[str]) -> np.ndarray:
        return full_pred if len(texts) == size else pipeline.predict(texts)

    return predict


def run_ablation(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cfg: dict[str, Any],
    best: dict[str, Any],
    seed: int,
    fractions: tuple[float, ...] = FRACTIONS,
) -> tuple[list[dict[str, Any]], pd.DataFrame | None]:
    """Treina um TF-IDF por fração do treino.

    Returns:
        ``(linhas da curva, previsões por exemplo do ponto de 100%)``. As previsões são
        ``None`` se ``fractions`` não incluir 1.0.
    """
    rows: list[dict[str, Any]] = []
    full_predictions: pd.DataFrame | None = None

    for fraction in fractions:
        n_train = len(train) if fraction >= 1 else round(len(train) * fraction)
        subset = stratified_subset(train, n_train, seed)

        started = time.perf_counter()
        features = build_features(cfg)
        x_train = features.fit_transform(subset["text"].tolist())
        classifier = LogisticRegression(
            C=float(best["C"]),
            class_weight=best["class_weight"],
            max_iter=int(cfg["max_iter"]),
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            classifier.fit(x_train, subset["label"].to_numpy())
        seconds = time.perf_counter() - started
        pipeline = Pipeline([("features", features), ("clf", classifier)])

        row: dict[str, Any] = {
            "fraction": fraction,
            "n_train": int(len(subset)),
            "train_seconds": round(seconds, 1),
            "converged": bool(int(classifier.n_iter_.max()) < int(cfg["max_iter"])),
        }
        preds: dict[str, np.ndarray] = {}
        for name, part in (("val", val), ("test", test)):
            texts = part["text"].tolist()
            preds[name] = pipeline.predict(texts)
            result = evaluate(
                _cached(pipeline, preds[name]),
                texts,
                part["label"].to_numpy(),
                LABELS,
                name=f"tfidf {fraction:.0%} ({name})",
                latency_samples=20,
            )
            row[name] = {
                "macro_f1": result["macro_f1"],
                "accuracy": result["accuracy"],
                "f1_per_class": result["f1_per_class"],
            }
        rows.append(row)
        print(
            f"{fraction:>5.0%} n={row['n_train']:6d} | val macro-F1={row['val']['macro_f1']:.4f} "
            f"| teste macro-F1={row['test']['macro_f1']:.4f} | {seconds:.0f}s"
        )

        if fraction >= 1:
            full_predictions = pd.concat(
                [
                    make_frame("val", val["row_id"], val["label"], preds["val"]),
                    make_frame("test", test["row_id"], test["label"], preds["test"]),
                ],
                ignore_index=True,
            )
    return rows, full_predictions


def main() -> None:
    config = load_config()
    seed = int(config["seed"])
    baseline = json.loads(Path(config["results_path"]).read_text(encoding="utf-8"))
    best = baseline["selection"]["best"]
    print(f"configuração do baseline: C={best['C']}, class_weight={best['class_weight']}")

    splits = load_splits(config)
    rows, predictions = run_ablation(
        splits["train"], splits["val"], splits["test"], config["tfidf"], best, seed
    )

    check: dict[str, Any] | None = None
    if predictions is not None:
        save_predictions(predictions, PREDICTIONS_PATH)
        full = next(r for r in rows if r["fraction"] >= 1)
        expected = float(baseline["test"]["macro_f1"])
        diff = abs(full["test"]["macro_f1"] - expected)
        check = {"baseline_test_macro_f1": expected, "abs_diff": diff, "reproduces": diff < 1e-3}
        print(f"\nconferência com o baseline: diferença de macro-F1 no teste = {diff:.2e}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, "config": best, "reproduces_baseline": check, "rows": rows}
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(f"curva em {RESULTS_PATH}; previsões em {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
