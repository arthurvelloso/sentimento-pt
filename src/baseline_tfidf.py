"""Baseline: TF-IDF (palavras e n-gramas de caracteres) + regressão logística.

Os hiperparâmetros (C e class_weight) são escolhidos na validação, por macro-F1.
O conjunto de teste é tocado uma única vez, no fim, pelo avaliador único
(``src.evaluate``), e o resultado vai para ``results/tfidf.json``.

Rode com ``make baseline``.
"""

from __future__ import annotations

import warnings
from itertools import product
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import load_config
from src.data import LABELS, load_splits
from src.evaluate import evaluate, save_result


def build_features(cfg: dict[str, Any]) -> FeatureUnion:
    """TF-IDF de palavras (1-2 gramas) junto com n-gramas de caracteres.

    Os caracteres pegam erros de digitação e variações ("otimo", "ótimo",
    "otimoo"), comuns em avaliações de clientes. Ajustado só no treino.
    """
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=tuple(cfg["word_ngram"]),
                    min_df=cfg["word_min_df"],
                    max_features=cfg["word_max_features"],
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=tuple(cfg["char_ngram"]),
                    min_df=cfg["char_min_df"],
                    max_features=cfg["char_max_features"],
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def fit_and_select(
    x_train: list[str],
    y_train: np.ndarray,
    x_val: list[str],
    y_val: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
) -> tuple[Pipeline, list[dict[str, Any]]]:
    """Treina uma regressão logística por combinação da grade e fica com a melhor na validação.

    O vetorizador é ajustado uma vez, só no treino. Só a regressão logística varia.

    Returns:
        ``(pipeline com a melhor combinação, registro da grade)``.
    """
    features = build_features(cfg)
    x_train_vec = features.fit_transform(x_train)
    x_val_vec = features.transform(x_val)

    grid = cfg["grid"]
    records: list[dict[str, Any]] = []
    best_clf, best_f1 = None, -1.0

    for c_value, class_weight in product(grid["C"], grid["class_weight"]):
        classifier = LogisticRegression(
            C=float(c_value),
            class_weight=class_weight,
            max_iter=int(cfg["max_iter"]),
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)  # registramos `converged` abaixo
            classifier.fit(x_train_vec, y_train)
        pred = classifier.predict(x_val_vec)
        macro_f1 = float(f1_score(y_val, pred, average="macro"))
        records.append(
            {
                "C": float(c_value),
                "class_weight": class_weight,
                "val_macro_f1": macro_f1,
                "val_accuracy": float(accuracy_score(y_val, pred)),
                "converged": bool(int(classifier.n_iter_.max()) < int(cfg["max_iter"])),
            }
        )
        print(
            f"C={c_value:<6g} class_weight={str(class_weight):9s} "
            f"val macro-F1={macro_f1:.4f}  acc={records[-1]['val_accuracy']:.4f}"
        )
        if macro_f1 > best_f1:
            best_clf, best_f1 = classifier, macro_f1

    assert best_clf is not None
    return Pipeline([("features", features), ("clf", best_clf)]), records


def run(config: dict[str, Any]) -> dict[str, Any]:
    cfg, seed = config["tfidf"], int(config["seed"])
    splits = load_splits(config)
    train, val, test = splits["train"], splits["val"], splits["test"]

    pipeline, records = fit_and_select(
        train["text"].tolist(),
        train["label"].to_numpy(),
        val["text"].tolist(),
        val["label"].to_numpy(),
        cfg,
        seed,
    )
    best = max(records, key=lambda r: r["val_macro_f1"])
    print(f"\nescolhido na validação: C={best['C']}, class_weight={best['class_weight']}")

    def predict_fn(texts: list[str]) -> np.ndarray:
        return pipeline.predict(texts)

    val_result = evaluate(
        predict_fn,
        val["text"].tolist(),
        val["label"].to_numpy(),
        LABELS,
        name="tfidf+logreg (validação)",
    )
    # O teste entra aqui, uma única vez, depois de fixar tudo na validação.
    test_result = evaluate(
        predict_fn,
        test["text"].tolist(),
        test["label"].to_numpy(),
        LABELS,
        name="tfidf+logreg",
    )

    result = {
        "model": "TF-IDF (palavras 1-2 + caracteres 2-5) + regressão logística",
        "seed": seed,
        "config": cfg,
        "n_features": int(pipeline.named_steps["features"].transform(["x"]).shape[1]),
        "selection": {"criterion": "macro-F1 na validação", "grid": records, "best": best},
        "val": val_result,
        "test": test_result,
    }
    save_result(result, config["results_path"])
    return result


def main() -> None:
    result = run(load_config())
    test = result["test"]
    print(
        f"\nTESTE: macro-F1={test['macro_f1']:.4f}  acurácia={test['accuracy']:.4f}  "
        f"latência p50={test['latency']['p50_ms']:.2f} ms  p95={test['latency']['p95_ms']:.2f} ms"
    )
    print("F1 por classe:", {k: round(v, 4) for k, v in test["f1_per_class"].items()})


if __name__ == "__main__":
    main()
