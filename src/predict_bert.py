"""Previsões por exemplo do BERTimbau ajustado, para a análise de erros.

Carrega o modelo (do Hugging Face Hub ou de uma pasta local), prevê validação e teste e
grava ``results/preds_bertimbau.csv`` (``split, row_id, y_true, y_pred`` e as
probabilidades por classe), sem o texto das avaliações.

O ``max_length`` precisa ser o mesmo do treino. Se ``results/bertimbau.json`` existir,
ele é lido de lá; senão é recalculado do mesmo jeito (percentil 95 dos tamanhos em tokens
do treino), que dá o mesmo valor.

Para um modelo privado no Hub, faça login antes (``hf auth login``) ou defina ``HF_TOKEN``.

    python -m src.predict_bert --model arthurvelloso/bertimbau-sentimento-b2w
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import load_config
from src.data import LABELS, load_splits
from src.predictions import make_frame, save_predictions
from src.train_bert import compute_max_length, predict_logits

DEFAULT_OUT = Path("results/preds_bertimbau.csv")
TRAIN_RESULTS = Path("results/bertimbau.json")


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax estável por linha."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def resolve_max_length(
    explicit: int | None,
    results_path: Path,
    tokenizer: Any,
    train_texts: Sequence[str],
    percentile: float,
) -> tuple[int, str]:
    """Descobre o ``max_length`` do treino. Devolve ``(valor, origem)``."""
    if explicit is not None:
        return int(explicit), "argumento"
    if results_path.exists():
        saved = json.loads(results_path.read_text(encoding="utf-8"))
        return int(saved["config"]["max_length"]), str(results_path)
    lengths = [len(ids) for ids in tokenizer(list(train_texts), truncation=False)["input_ids"]]
    return compute_max_length(lengths, percentile), "recalculado do treino"


def predict_frame(
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    split: str,
    max_length: int,
    batch_size: int = 128,
) -> pd.DataFrame:
    """Previsões (classe e probabilidades) de um split, no formato de ``predictions``."""
    logits = predict_logits(model, tokenizer, frame["text"].tolist(), max_length, batch_size)
    proba = softmax(logits)
    return make_frame(
        split,
        frame["row_id"],
        frame["label"],
        proba.argmax(axis=-1),
        proba=proba,
        labels=LABELS,
    )


def check_against_training(frame: pd.DataFrame, results_path: Path) -> dict[str, Any] | None:
    """Compara o macro-F1 do teste recalculado com o registrado no treino.

    Um desvio grande indica modelo, ``max_length`` ou splits diferentes dos do treino.
    """
    if not results_path.exists():
        return None
    saved = json.loads(results_path.read_text(encoding="utf-8"))["test"]
    test = frame[frame["split"] == "test"]
    macro_f1 = float(f1_score(test["y_true"], test["y_pred"], average="macro"))
    diff = abs(macro_f1 - float(saved["macro_f1"]))
    return {
        "macro_f1_recalculado": macro_f1,
        "macro_f1_do_treino": float(saved["macro_f1"]),
        "diferenca": diff,
        "confere": diff < 5e-3,  # fp16 vs fp32 pode mudar algumas previsões
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="repo do Hub (usuario/nome) ou pasta local")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--max-length", type=int, dest="max_length")
    parser.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    return parser.parse_args(argv)


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    args = parse_args()
    config = load_config()
    splits = load_splits(config)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)
    print(f"modelo {args.model} em {device}")

    max_length, origin = resolve_max_length(
        args.max_length,
        TRAIN_RESULTS,
        tokenizer,
        splits["train"]["text"],
        float(config["bert"]["max_length_percentile"]),
    )
    print(f"max_length={max_length} ({origin})")

    parts = [
        predict_frame(model, tokenizer, splits[name], name, max_length, args.batch_size)
        for name in args.splits
    ]
    frame = pd.concat(parts, ignore_index=True)
    save_predictions(frame, args.out)

    for name in args.splits:
        part = frame[frame["split"] == name]
        print(
            f"{name:5s} macro-F1={f1_score(part['y_true'], part['y_pred'], average='macro'):.4f} "
            f"acurácia={accuracy_score(part['y_true'], part['y_pred']):.4f}"
        )
    check = check_against_training(frame, TRAIN_RESULTS)
    if check is None:
        print("results/bertimbau.json não encontrado: sem conferência com o treino.")
    else:
        status = "CONFERE" if check["confere"] else "NÃO CONFERE"
        print(
            f"conferência com o treino ({status}): {check['macro_f1_recalculado']:.4f} "
            f"vs {check['macro_f1_do_treino']:.4f}"
        )
    print(f"previsões em {args.out}")


if __name__ == "__main__":
    main()
