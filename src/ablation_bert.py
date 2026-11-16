"""Agrega a ablação do tamanho dos dados do BERTimbau.

Cada tamanho é um treino separado (``python -m src.train_bert --train-subset N``, na GPU),
que grava ``results/ablation_bert/n{N}.json``. Este script reúne esses arquivos numa curva
com o mesmo formato da do TF-IDF (``results/ablation_tfidf.json``).

O ponto de 100% vem das previsões do modelo já treinado (``results/preds_bertimbau.csv``),
recalculadas aqui: assim a curva não depende do JSON do treino principal.

Roda só com CPU, sem torch: ``python -m src.ablation_bert``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.analysis import split_metrics
from src.predictions import load_predictions

SUBSET_DIR = Path("results/ablation_bert")
PREDICTIONS_PATH = Path("results/preds_bertimbau.csv")
OUTPUT_PATH = Path("results/ablation_bertimbau.json")
META_PATH = Path("splits/meta.json")


def _slim(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro_f1": float(result["macro_f1"]),
        "accuracy": float(result["accuracy"]),
        "f1_per_class": result["f1_per_class"],
    }


def collect_rows(
    subset_dir: Path, predictions_path: Path, n_train_full: int
) -> list[dict[str, Any]]:
    """Linhas da curva, ordenadas por tamanho de treino."""
    rows: list[dict[str, Any]] = []
    for path in sorted(subset_dir.glob("n*.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        n_train = int(saved["config"]["n_train"])
        rows.append(
            {
                "fraction": round(n_train / n_train_full, 4),
                "n_train": n_train,
                "train_seconds": saved["training"]["train_seconds"],
                "best_epoch": saved["training"]["best_epoch"],
                "val": _slim(saved["val"]),
                "test": _slim(saved["test"]),
            }
        )

    if predictions_path.exists():
        row: dict[str, Any] = {"fraction": 1.0, "n_train": n_train_full, "source": "previsões"}
        for split in ("val", "test"):
            frame = load_predictions(predictions_path, split=split)
            metrics = split_metrics(frame["y_true"], frame["y_pred"])
            row[split] = {k: metrics[k] for k in ("macro_f1", "accuracy", "f1_per_class")}
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"nada para agregar: sem {subset_dir}/n*.json e sem {predictions_path}"
        )
    return sorted(rows, key=lambda r: r["n_train"])


def main() -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    n_full = int(meta["splits"]["train"]["n"])
    rows = collect_rows(SUBSET_DIR, PREDICTIONS_PATH, n_full)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for row in rows:
        val, test = row["val"]["macro_f1"], row["test"]["macro_f1"]
        print(
            f"{row['fraction']:>6.0%} n={row['n_train']:6d} | "
            f"val macro-F1={val:.4f} | teste macro-F1={test:.4f}"
        )
    print(f"curva em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
