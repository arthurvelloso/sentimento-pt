"""Fine-tuning do LLM com QLoRA para classificar o sentimento.

O modelo base fica congelado em 4 bits (NF4) e só um adaptador LoRA é treinado, sobre as
camadas lineares da atenção e do MLP. O exemplo de treino é exatamente o prompt zero-shot
de ``src.llm_classify`` seguido da resposta (``negativo``, ``neutro`` ou ``positivo``), e a
loss é calculada só nos tokens da resposta. Assim a avaliação reaproveita o mesmo prompt,
o mesmo parser e o mesmo ``src.evaluate`` do zero-shot, e a única diferença entre as duas
linhas da tabela é o adaptador.

Treino (GPU; ver ``notebooks/qlora_train.ipynb``)::

    python -m src.qlora
    python -m src.qlora --max-steps 20      # teste rápido de ponta a ponta

Avaliação, depois do treino::

    python -m src.llm_classify --split val --max-eval 600 --adapter outputs/qlora/adapter
    python -m src.llm_classify --split test --adapter outputs/qlora/adapter
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_config
from src.data import LABELS, load_splits, stratified_subset
from src.llm_classify import build_messages, render_prompts

IGNORE_INDEX = -100  # posições que a loss ignora (convenção do PyTorch e do transformers)

# ------------------------------------------------------------ dados (sem torch)


def build_example(tokenizer: Any, text: str, label: int, max_chars: int) -> dict[str, list[int]]:
    """Tokeniza prompt + resposta. ``labels`` só tem valor nos tokens da resposta.

    O prompt é renderizado como na inferência (``add_generation_prompt``), e a resposta
    termina com o token de fim de turno, para o modelo aprender a parar.
    """
    prompt = render_prompts(tokenizer, [build_messages(text, [], max_chars)])[0]
    answer = LABELS[label] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + answer_ids,
    }


def collate(examples: Sequence[dict[str, list[int]]], pad_id: int) -> dict[str, list[list[int]]]:
    """Junta exemplos num batch, com padding à direita (labels do padding = ignorado)."""
    width = max(len(e["input_ids"]) for e in examples)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for e in examples:
        pad = width - len(e["input_ids"])
        batch["input_ids"].append(e["input_ids"] + [pad_id] * pad)
        batch["attention_mask"].append([1] * len(e["input_ids"]) + [0] * pad)
        batch["labels"].append(e["labels"] + [IGNORE_INDEX] * pad)
    return batch


def plan_steps(
    n_examples: int, micro_batch: int, grad_accum: int, epochs: int, seed: int
) -> list[list[list[int]]]:
    """Plano do treino: uma lista de passos; cada passo tem ``grad_accum`` micro-batches de índices.

    Cada época embaralha com ``seed + época``. Os últimos micro-batches que sobram formam um
    passo menor, em vez de serem descartados.
    """
    steps: list[list[list[int]]] = []
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(n_examples).tolist()
        micro = [order[i : i + micro_batch] for i in range(0, n_examples, micro_batch)]
        steps += [micro[i : i + grad_accum] for i in range(0, len(micro), grad_accum)]
    return steps


def learning_rate_at(step: int, total: int, base_lr: float, warmup_ratio: float) -> float:
    """Aquecimento linear até ``base_lr`` e queda linear até 0. ``step`` começa em 0."""
    warmup = max(1, math.ceil(total * warmup_ratio))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr * max(0.0, (total - step) / max(1, total - warmup))


# ------------------------------------------------------------------ modelo (torch)


def quantization_config() -> Any:
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # a T4 não tem bf16 nativo
    )


def load_quantized(name: str) -> tuple[Any, Any]:
    """Modelo base em 4 bits na GPU 0 e seu tokenizador."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA precisa de GPU CUDA (bitsandbytes).")
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=quantization_config(), dtype=torch.float16, device_map={"": 0}
    )
    return model, tokenizer


def load_with_adapter(name: str, adapter: str) -> tuple[Any, Any]:
    """Base em 4 bits (como no treino) com o adaptador LoRA carregado, para inferência."""
    from peft import PeftModel

    model, tokenizer = load_quantized(name)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def lora_config(cfg: dict[str, Any]) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=list(cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )


def train(
    model: Any, examples: list[dict[str, list[int]]], cfg: dict[str, Any], pad_id: int, seed: int
) -> dict[str, Any]:
    """Laço de treino: fp16 com GradScaler, acumulação de gradiente e clipping."""
    import torch

    device = next(model.parameters()).device
    steps = plan_steps(
        len(examples),
        int(cfg["micro_batch_size"]),
        int(cfg["grad_accum"]),
        int(cfg["epochs"]),
        seed,
    )
    if cfg.get("max_steps"):
        steps = steps[: int(cfg["max_steps"])]
    total = len(steps)
    base_lr = float(cfg["learning_rate"])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()

    model.train()
    curve: list[dict[str, float]] = []
    window_loss, window_tokens, window_steps = 0.0, 0, 0
    started = time.perf_counter()
    window_started = started
    for step, micro_batches in enumerate(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step, total, base_lr, float(cfg["warmup_ratio"]))
        step_loss = 0.0
        for indices in micro_batches:
            batch = collate([examples[i] for i in indices], pad_id)
            tensors = {k: torch.tensor(v, device=device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**tensors, use_cache=False).loss
            scaler.scale(loss / len(micro_batches)).backward()
            step_loss += loss.item() / len(micro_batches)
            window_tokens += int(tensors["attention_mask"].sum())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        window_loss += step_loss
        window_steps += 1
        if window_steps == int(cfg["log_every"]) or step + 1 == total:
            now = time.perf_counter()
            point = {
                "step": step + 1,
                "loss": window_loss / window_steps,
                "tokens_per_s": window_tokens / (now - window_started),
                "elapsed_s": now - started,
            }
            curve.append(point)
            remaining = (now - started) / (step + 1) * (total - step - 1)
            print(
                f"passo {point['step']}/{total}  loss={point['loss']:.4f}  "
                f"{point['tokens_per_s']:.0f} tok/s  restam ~{remaining / 60:.0f} min",
                flush=True,
            )
            window_loss, window_tokens, window_steps, window_started = 0.0, 0, 0, now
    return {
        "steps": total,
        "seconds": round(time.perf_counter() - started, 1),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "loss_curve": curve,
    }


def run(config: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    import peft
    import torch
    import transformers
    from peft import get_peft_model, prepare_model_for_kbit_training

    cfg = dict(config["qlora"]) | {k: v for k, v in options.items() if v is not None}
    seed = int(config["seed"])
    model_name = config["llm"]["model_name"]
    max_chars = int(config["llm"]["max_review_chars"])

    frame = stratified_subset(load_splits(config)["train"], cfg["train_subset"], seed)
    print(f"{model_name} + QLoRA | treino: {len(frame)} exemplos | seed {seed}")

    model, tokenizer = load_quantized(model_name)
    examples = [
        build_example(tokenizer, text, int(label), max_chars)
        for text, label in zip(frame["text"], frame["label"], strict=True)
    ]
    lengths = np.array([len(e["input_ids"]) for e in examples])
    print(f"tokens por exemplo: média {lengths.mean():.0f}, máximo {lengths.max()}")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_config(cfg))
    trainable, total = model.get_nb_trainable_parameters()
    print(f"parâmetros treináveis: {trainable:,} de {total:,} ({trainable / total:.2%})")

    stats = train(model, examples, cfg, tokenizer.pad_token_id, seed)

    out_dir = Path(cfg["output_dir"]) / "adapter"
    model.save_pretrained(out_dir)
    print(f"adaptador salvo em {out_dir}")
    if cfg.get("hub_repo"):
        model.push_to_hub(cfg["hub_repo"], private=True)
        print(f"adaptador enviado a {cfg['hub_repo']} (privado)")

    result = {
        "model": model_name,
        "method": "QLoRA (NF4, double quant) com loss só na resposta",
        "prompt": "zero-shot v1 de src.llm_classify",
        "seed": seed,
        "train_examples": len(frame),
        "tokens_per_example": {"mean": float(lengths.mean()), "max": int(lengths.max())},
        "hyperparameters": {
            k: cfg[k]
            for k in (
                "epochs",
                "learning_rate",
                "micro_batch_size",
                "grad_accum",
                "warmup_ratio",
                "max_grad_norm",
                "lora_r",
                "lora_alpha",
                "lora_dropout",
                "target_modules",
            )
        }
        | {"max_steps": cfg.get("max_steps")},
        "trainable_parameters": trainable,
        "total_parameters": total,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    } | stats
    path = Path(cfg["results_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"resumo do treino em {path}")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--max-steps", type=int, dest="max_steps", help="para no passo N (teste)")
    parser.add_argument("--train-subset", type=int, dest="train_subset")
    parser.add_argument("--hub-repo", dest="hub_repo", help='"usuario/nome": envia o adaptador')
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument("--results-path", dest="results_path")
    return parser.parse_args(argv)


def main() -> None:
    result = run(load_config(), vars(parse_args()))
    print(
        f"\nfim: {result['steps']} passos em {result['seconds'] / 60:.1f} min, "
        f"pico de {result['peak_gpu_memory_gb']} GB de GPU"
    )


if __name__ == "__main__":
    main()
