"""Fine-tuning do BERTimbau para classificação de sentimento.

Usa os mesmos splits e o mesmo avaliador (``src.evaluate``) do baseline TF-IDF,
então os números são diretamente comparáveis. O conjunto de teste é tocado uma
única vez, depois de escolher o melhor checkpoint pela validação (macro-F1).

Pensado para rodar em GPU grátis (Kaggle ou Colab): ver ``notebooks/bertimbau_train.ipynb``.
Também roda em CPU ou MPS, devagar, e é o que os testes usam com um modelo minúsculo.

    python -m src.train_bert --hub-repo SEU_USUARIO/bertimbau-sentimento-b2w

``torch`` e ``transformers`` são importados dentro das funções que precisam deles,
para o resto do projeto (e a CI) funcionar sem essas dependências pesadas.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data import LABELS, load_splits
from src.evaluate import evaluate, save_result

# ----------------------------------------------------------------- utilitários


def compute_max_length(
    token_lengths: Sequence[int],
    percentile: float = 95.0,
    *,
    minimum: int = 16,
    maximum: int = 512,
    multiple: int = 8,
) -> int:
    """Comprimento máximo em tokens que cobre ``percentile``% dos textos.

    Textos acima disso são truncados. Padding até o fim do batch gasta memória e
    tempo; truncar os 5% mais longos custa pouco em qualidade. O resultado é
    arredondado para cima a um múltiplo de ``multiple`` (melhor uso da GPU) e
    limitado a ``[minimum, maximum]`` (512 é o limite do BERT).
    """
    lengths = np.asarray(token_lengths)
    if lengths.size == 0:
        raise ValueError("sem comprimentos para calcular o percentil")
    value = int(np.ceil(np.percentile(lengths, percentile)))
    rounded = int(np.ceil(value / multiple) * multiple)
    return max(minimum, min(rounded, maximum))


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Métricas por época, usadas pelo Trainer para escolher o melhor checkpoint."""
    predictions = eval_pred.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    labels = np.asarray(eval_pred.label_ids)
    predicted = np.argmax(predictions, axis=-1)
    return {
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "accuracy": float(accuracy_score(labels, predicted)),
    }


def stratified_subset(frame: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Amostra ``n`` linhas mantendo a proporção das classes. ``None`` devolve tudo."""
    if n is None or n >= len(frame):
        return frame
    if n < len(np.unique(frame["label"])):
        raise ValueError("subconjunto menor que o número de classes")
    sampled, _ = train_test_split(frame, train_size=n, stratify=frame["label"], random_state=seed)
    return sampled.reset_index(drop=True)


def resolve_results_path(config: dict[str, Any], seed: int, override: str | None) -> Path:
    """Caminho do JSON de resultados; seeds extras ganham sufixo para não sobrescrever."""
    if override:
        return Path(override)
    base = Path(config["bert"]["results_path"])
    if seed == int(config["seed"]):
        return base
    return base.with_name(f"{base.stem}_seed{seed}{base.suffix}")


class TokenizedDataset:
    """Dataset em memória de textos já tokenizados, sem padding.

    O padding é feito por batch (``DataCollatorWithPadding``), então batches de
    textos curtos ficam curtos.
    """

    def __init__(self, encodings: dict[str, list[list[int]]], labels: Sequence[int]) -> None:
        if len(encodings["input_ids"]) != len(labels):
            raise ValueError("textos e rótulos com tamanhos diferentes")
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


def tokenize(tokenizer: Any, texts: Sequence[str], max_length: int) -> dict[str, list[list[int]]]:
    """Tokeniza com truncamento em ``max_length`` e sem padding."""
    encoded = tokenizer(
        list(texts),
        truncation=True,
        max_length=max_length,
        padding=False,
        return_token_type_ids=False,
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


# ------------------------------------------------------------ predição / treino


def predict_logits(
    model: Any, tokenizer: Any, texts: Sequence[str], max_length: int, batch_size: int = 128
) -> np.ndarray:
    """Logits para ``texts``, na ordem original.

    Ordena por tamanho antes de formar os batches (menos padding, bem mais rápido)
    e depois restaura a ordem.
    """
    import torch

    texts = list(texts)
    if not texts:
        return np.empty((0, model.config.num_labels))
    device = next(model.parameters()).device
    model.eval()

    order = np.argsort([len(text) for text in texts], kind="stable")
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = [texts[i] for i in order[start : start + batch_size]]
        batch = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(device)
        with torch.inference_mode():
            chunks.append(model(**batch).logits.float().cpu().numpy())

    sorted_logits = np.concatenate(chunks)
    logits = np.empty_like(sorted_logits)
    logits[order] = sorted_logits
    return logits


def make_predict_fn(model: Any, tokenizer: Any, max_length: int, batch_size: int = 128) -> Any:
    """Adapta o modelo à interface do avaliador: lista de textos -> classes."""

    def predict(texts: list[str]) -> np.ndarray:
        return predict_logits(model, tokenizer, texts, max_length, batch_size).argmax(axis=-1)

    return predict


def build_training_args(
    hp: dict[str, Any], output_dir: Path, seed: int, hub_repo: str | None
) -> Any:
    """TrainingArguments: avalia e salva a cada época, e fica com o melhor por macro-F1."""
    import torch
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=hp["epochs"],
        learning_rate=hp["learning_rate"],
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["eval_batch_size"],
        weight_decay=hp["weight_decay"],
        lr_scheduler_type="linear",
        # transformers 5: float em [0, 1) é a fração dos passos usada no warmup.
        warmup_steps=hp["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available() and not hp.get("use_cpu", False),
        use_cpu=bool(hp.get("use_cpu", False)),
        seed=seed,
        data_seed=seed,
        logging_steps=100,
        report_to="none",
        remove_unused_columns=False,
        push_to_hub=bool(hub_repo),
        hub_model_id=hub_repo,
        hub_private_repo=True if hub_repo else None,
        # Sobe o modelo a cada época: se a sessão de GPU cair, nada se perde.
        hub_strategy="every_save",
    )


def fine_tune(
    model: Any,
    tokenizer: Any,
    train_ds: TokenizedDataset,
    val_ds: TokenizedDataset,
    hp: dict[str, Any],
    *,
    output_dir: Path,
    seed: int,
    hub_repo: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Treina e devolve ``(trainer, informações do treino)``.

    Ao fim, ``trainer.model`` é o checkpoint da melhor época na validação.
    """
    from transformers import DataCollatorWithPadding, Trainer, set_seed

    set_seed(seed)
    trainer = Trainer(
        model=model,
        args=build_training_args(hp, output_dir, seed, hub_repo),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    start = time.perf_counter()
    train_output = trainer.train()
    seconds = time.perf_counter() - start

    history = [
        {
            "epoch": float(entry["epoch"]),
            "val_macro_f1": float(entry["eval_macro_f1"]),
            "val_accuracy": float(entry["eval_accuracy"]),
            "val_loss": float(entry["eval_loss"]),
        }
        for entry in trainer.state.log_history
        if "eval_macro_f1" in entry
    ]
    best_metric = trainer.state.best_metric
    best_epoch = next(
        (
            h["epoch"]
            for h in history
            if best_metric is not None and h["val_macro_f1"] == best_metric
        ),
        None,
    )
    info = {
        "train_seconds": round(seconds, 1),
        "train_samples_per_second": float(train_output.metrics.get("train_samples_per_second", 0)),
        "final_train_loss": float(train_output.training_loss),
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_metric) if best_metric is not None else None,
        "history": history,
    }
    return trainer, info


def describe_environment(model: Any) -> dict[str, Any]:
    """Versões e hardware, para o resultado ser reproduzível e interpretável."""
    import torch
    import transformers

    device = next(model.parameters()).device
    if torch.cuda.is_available() and device.type == "cuda":
        gpu = torch.cuda.get_device_name(0)
    elif device.type == "mps":
        gpu = f"Apple MPS ({platform.machine()})"
    else:
        gpu = "cpu"
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device.type,
        "gpu": gpu,
        "fp16": bool(torch.cuda.is_available() and device.type == "cuda"),
        "platform": platform.platform(),
    }


# ------------------------------------------------------------------- execução


def run(config: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    hp = dict(config["bert"]) | {
        k: v for k, v in options.items() if v is not None and k in config["bert"]
    }
    seed = int(options.get("seed") or config["seed"])
    hub_repo = options.get("hub_repo") or config["bert"].get("hub_repo")
    results_path = resolve_results_path(config, seed, options.get("results_path"))
    output_dir = Path(options.get("output_dir") or hp["output_dir"])
    hp["use_cpu"] = bool(options.get("use_cpu"))
    max_eval = options.get("max_eval")

    splits = load_splits(config)
    train = stratified_subset(splits["train"], hp.get("train_subset"), seed)
    val = stratified_subset(splits["val"], max_eval, seed)
    test = stratified_subset(splits["test"], max_eval, seed)
    is_subset = len(train) < len(splits["train"]) or len(test) < len(splits["test"])
    print(f"treino={len(train)} validação={len(val)} teste={len(test)} subset={is_subset}")

    tokenizer = AutoTokenizer.from_pretrained(hp["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        hp["model_name"],
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )

    lengths = [len(ids) for ids in tokenizer(train["text"].tolist(), truncation=False)["input_ids"]]
    max_length = compute_max_length(lengths, hp["max_length_percentile"])
    truncated = float(np.mean(np.asarray(lengths) > max_length))
    print(f"max_length={max_length} (p{hp['max_length_percentile']}); truncados: {truncated:.1%}")

    train_ds = TokenizedDataset(tokenize(tokenizer, train["text"], max_length), train["label"])
    val_ds = TokenizedDataset(tokenize(tokenizer, val["text"], max_length), val["label"])

    trainer, info = fine_tune(
        model,
        tokenizer,
        train_ds,
        val_ds,
        hp,
        output_dir=output_dir,
        seed=seed,
        hub_repo=hub_repo,
    )
    model = trainer.model  # melhor época na validação

    predict_fn = make_predict_fn(model, tokenizer, max_length, int(hp["eval_batch_size"]))
    val_result = evaluate(
        predict_fn,
        val["text"].tolist(),
        val["label"].to_numpy(),
        LABELS,
        name="bertimbau (validação)",
    )
    # O teste entra aqui, uma única vez, com o melhor checkpoint já escolhido.
    test_result = evaluate(
        predict_fn, test["text"].tolist(), test["label"].to_numpy(), LABELS, name="bertimbau"
    )

    result = {
        "model": f"{hp['model_name']} (fine-tuning completo)",
        "seed": seed,
        "is_subset": is_subset,
        "config": {
            "learning_rate": hp["learning_rate"],
            "epochs": hp["epochs"],
            "batch_size": hp["batch_size"],
            "warmup_ratio": hp["warmup_ratio"],
            "weight_decay": hp["weight_decay"],
            "max_length": max_length,
            "max_length_percentile": hp["max_length_percentile"],
            "truncated_share": truncated,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        },
        "environment": describe_environment(model),
        "training": info,
        "hub": {"repo": hub_repo, "private": True} if hub_repo else None,
        "val": val_result,
        "test": test_result,
    }
    save_result(result, results_path)
    print(f"resultado em {results_path}")

    if hub_repo:
        trainer.push_to_hub(commit_message="BERTimbau ajustado: melhor época pela validação")
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(results_path),
            path_in_repo=f"results/{results_path.name}",
            repo_id=hub_repo,
            repo_type="model",
        )
        print(f"modelo e resultados enviados para {hub_repo} (privado)")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, help="seed (padrão: a da config)")
    parser.add_argument("--epochs", type=int, help="número de épocas")
    parser.add_argument("--learning-rate", type=float, dest="learning_rate")
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--train-subset", type=int, dest="train_subset", help="limita o treino")
    parser.add_argument(
        "--max-eval", type=int, dest="max_eval", help="limita val e teste (só para fumaça)"
    )
    parser.add_argument("--hub-repo", dest="hub_repo", help="usuario/nome do repo privado no Hub")
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument("--results-path", dest="results_path")
    parser.add_argument("--use-cpu", action="store_true", dest="use_cpu")
    return parser.parse_args(argv)


def main() -> None:
    options = vars(parse_args())
    result = run(load_config(), options)
    test = result["test"]
    print(
        f"\nTESTE: macro-F1={test['macro_f1']:.4f}  acurácia={test['accuracy']:.4f}  "
        f"latência p50={test['latency']['p50_ms']:.1f} ms  p95={test['latency']['p95_ms']:.1f} ms"
    )
    print("F1 por classe:", {k: round(v, 4) for k, v in test["f1_per_class"].items()})


if __name__ == "__main__":
    main()
