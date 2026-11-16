"""Classificação de sentimento com um LLM aberto, em zero-shot ou few-shot.

O modelo recebe a avaliação num prompt fixo, responde com uma palavra (negativo, neutro
ou positivo) e um parser transforma a resposta em classe. Respostas que não são um
rótulo viram ``-1`` e contam como erro: a taxa de respostas inválidas faz parte do
resultado, não é escondida.

Avaliação com o mesmo ``src.evaluate`` do TF-IDF e do BERT. Decodificação gananciosa
(sem sorteio), então a mesma entrada dá sempre a mesma saída.

O prompt foi ajustado olhando só a validação (``--split val``, gravando em
``results/dev/``). O teste entra uma única vez, com o prompt já congelado.

    python -m src.llm_classify --model Qwen/Qwen3.5-4B --shots-per-class 0
    python -m src.llm_classify --model Qwen/Qwen3.5-4B --shots-per-class 3
"""

from __future__ import annotations

import argparse
import platform
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import LABELS, load_splits, normalize_text, stratified_subset
from src.evaluate import evaluate, save_result
from src.predictions import make_frame, save_predictions

PROMPT_VERSION = "v1"
INVALID = -1

SYSTEM_PROMPT = (
    "Você classifica o sentimento de avaliações de produtos de uma loja online brasileira. "
    "Responda com exatamente uma palavra: negativo, neutro ou positivo.\n"
    "- negativo: o cliente está insatisfeito (equivale às notas 1 ou 2 de 5).\n"
    "- neutro: opinião mista ou indiferente (equivale à nota 3 de 5).\n"
    "- positivo: o cliente está satisfeito (equivale às notas 4 ou 5 de 5)."
)

_STEMS = (("negativ", 0), ("neutr", 1), ("positiv", 2))
_NEGATIONS = {"nao", "nem", "sem", "nunca"}

# --------------------------------------------------------------------- prompt


def truncate(text: str, max_chars: int) -> str:
    """Corta avaliações muito longas, para o prompt não crescer sem limite."""
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


def format_review(text: str, max_chars: int) -> str:
    return f"Avaliação: {truncate(text, max_chars)}\nSentimento:"


def select_shots(
    train: pd.DataFrame, per_class: int, seed: int, max_chars: int = 300
) -> list[dict[str, Any]]:
    """Escolhe ``per_class`` exemplos de cada classe do treino, curtos e em ordem misturada.

    Só entram textos de até ``max_chars`` (prompts menores). A ordem final é embaralhada
    com a seed, para as classes não virem em bloco (o que enviesaria o modelo).
    """
    if per_class <= 0:
        return []
    rng = np.random.default_rng(seed)
    chosen: list[dict[str, Any]] = []
    for label in range(len(LABELS)):
        pool = train[(train["label"] == label) & (train["text"].str.len() <= max_chars)]
        if len(pool) < per_class:
            raise ValueError(f"poucos exemplos curtos da classe {LABELS[label]!r} para o few-shot")
        picked = pool.iloc[rng.choice(len(pool), size=per_class, replace=False)]
        chosen += [
            {"row_id": int(r.row_id), "text": r.text, "label": int(r.label)}
            for r in picked.itertuples()
        ]
    order = rng.permutation(len(chosen))
    return [chosen[i] for i in order]


def build_messages(text: str, shots: Sequence[dict[str, Any]], max_chars: int) -> list[dict]:
    """Conversa: instrução, exemplos (usuário/assistente) e a avaliação a classificar."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot in shots:
        messages.append({"role": "user", "content": format_review(shot["text"], max_chars)})
        messages.append({"role": "assistant", "content": LABELS[shot["label"]]})
    messages.append({"role": "user", "content": format_review(text, max_chars)})
    return messages


# --------------------------------------------------------------------- parser


def _label_of(token: str) -> int | None:
    for stem, label in _STEMS:
        if token.startswith(stem):
            return label
    return None


def parse_answer(response: str) -> int:
    """Resposta do modelo -> classe (0, 1, 2) ou ``INVALID``.

    Regras, nesta ordem:
    1. A primeira palavra é um rótulo (aceita "positiva", "negativa", maiúsculas, acento
       e pontuação): essa é a classe.
    2. Senão, se a resposta tem negação ("não", "sem"...), é inválida: "não é positivo"
       não pode virar positivo.
    3. Senão, se exatamente um rótulo aparece na resposta, é ele.
    4. Qualquer outra coisa é inválida.
    """
    tokens = normalize_text(response).split()
    if not tokens:
        return INVALID
    first = _label_of(tokens[0])
    if first is not None:
        return first
    if any(token in _NEGATIONS for token in tokens):
        return INVALID
    found = {label for token in tokens if (label := _label_of(token)) is not None}
    return found.pop() if len(found) == 1 else INVALID


# ------------------------------------------------------------------- geração


def render_prompts(tokenizer: Any, conversations: Sequence[list[dict]]) -> list[str]:
    """Aplica o template de chat do modelo, com o modo de raciocínio desligado."""
    return [
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for messages in conversations
    ]


def generate_answers(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    batch_size: int = 16,
    max_new_tokens: int = 6,
) -> list[str]:
    """Respostas do modelo, na ordem dos prompts (decodificação gananciosa).

    Ordena por tamanho antes de formar os batches (menos padding) e restaura a ordem.
    """
    import torch

    if not prompts:
        return []
    tokenizer.padding_side = "left"  # geração exige padding à esquerda
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    model.eval()

    order = np.argsort([len(p) for p in prompts], kind="stable")
    answers: list[str] = [""] * len(prompts)
    for start in range(0, len(prompts), batch_size):
        indices = order[start : start + batch_size]
        batch = tokenizer(
            [prompts[i] for i in indices],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        with torch.inference_mode():
            output = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = output[:, batch["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for index, text in zip(indices, decoded, strict=True):
            answers[int(index)] = text.strip()
    return answers


def classify(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    shots: Sequence[dict[str, Any]],
    *,
    max_chars: int,
    batch_size: int,
    max_new_tokens: int,
) -> tuple[np.ndarray, list[str]]:
    """Classes previstas (``-1`` = inválida) e as respostas cruas do modelo."""
    conversations = [build_messages(t, shots, max_chars) for t in texts]
    prompts = render_prompts(tokenizer, conversations)
    answers = generate_answers(
        model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens
    )
    return np.array([parse_answer(a) for a in answers], dtype=int), answers


# ------------------------------------------------------------------- execução


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_llm(name: str, dtype: str | None = None) -> tuple[Any, Any, str]:
    """Carrega tokenizador e modelo na melhor opção disponível. Devolve ``(modelo, tok, dtype)``."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device, default_dtype = "cuda", "float16"  # T4 não tem bf16 nativo
    elif torch.backends.mps.is_available():
        device, default_dtype = "mps", "float16"
    else:
        device, default_dtype = "cpu", "float32"
    dtype = dtype or default_dtype

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype))
    model.to(device)
    return model, tokenizer, dtype


def describe_environment(model: Any, dtype: str) -> dict[str, Any]:
    import torch
    import transformers

    device = next(model.parameters()).device
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else device.type
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device.type,
        "gpu": gpu,
        "dtype": dtype,
        "platform": platform.platform(),
    }


def resolve_eval_subset(max_eval: int | None, split: str, cfg: dict[str, Any]) -> int | None:
    """Tamanho do subconjunto avaliado: o pedido na linha de comando, senão o da config.

    ``None`` na config significa o split inteiro. Atenção: o argparse devolve ``None`` para
    uma flag não passada, então não dá para usar ``dict.get(chave, padrão)`` aqui (a chave
    existe, com valor ``None``).
    """
    if max_eval is not None:
        return int(max_eval)
    default = cfg["dev_subset"] if split == "val" else cfg["eval_subset"]
    return None if default is None else int(default)


def run(config: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config["llm"]) | {
        k: v for k, v in options.items() if v is not None and k in config["llm"]
    }
    seed = int(config["seed"])
    split = options.get("split") or "test"
    shots_per_class = int(cfg["shots_per_class"])
    subset = resolve_eval_subset(options.get("max_eval"), split, cfg)

    splits = load_splits(config)
    evaluation = stratified_subset(splits[split], subset, seed)
    shots = select_shots(splits["train"], shots_per_class, seed)
    print(
        f"{cfg['model_name']} | {split}: {len(evaluation)} avaliações | "
        f"{shots_per_class} exemplo(s) por classe | prompt {PROMPT_VERSION}"
    )

    model, tokenizer, dtype = load_llm(cfg["model_name"], options.get("dtype"))
    settings = {
        "max_chars": int(cfg["max_review_chars"]),
        "batch_size": int(cfg["batch_size"]),
        "max_new_tokens": int(cfg["max_new_tokens"]),
    }

    captured: dict[str, Any] = {}
    n_eval = len(evaluation)

    def predict_fn(texts: list[str]) -> np.ndarray:
        preds, answers = classify(model, tokenizer, texts, shots, **settings)
        if len(texts) == n_eval:  # a passada completa; as chamadas de latência são de 1 texto
            captured["preds"], captured["answers"] = preds, answers
        return preds

    started = time.perf_counter()
    kind = f"{len(shots)}-shot" if shots else "zero-shot"
    name = f"{cfg['model_name']} {kind}"
    result = evaluate(
        predict_fn,
        evaluation["text"].tolist(),
        evaluation["label"].to_numpy(),
        LABELS,
        name=name,
        latency_samples=int(options.get("latency_samples") or 100),
    )
    seconds = time.perf_counter() - started

    preds, answers = captured["preds"], captured["answers"]
    invalid_mask = preds == INVALID
    result.update(
        {
            "model": cfg["model_name"],
            "split": split,
            "shots_per_class": shots_per_class,
            "prompt_version": PROMPT_VERSION,
            "prompt": {"system": SYSTEM_PROMPT, "shots": shots},
            "seed": seed,
            "invalid_share": float(invalid_mask.mean()),
            "invalid_examples": sorted(
                {a for a, bad in zip(answers, invalid_mask, strict=True) if bad}
            )[:20],
            "subset": {
                "n": n_eval,
                "of": int(len(splits[split])),
                "stratified": n_eval < len(splits[split]),
            },
            "settings": settings,
            "environment": describe_environment(model, dtype),
            "eval_seconds": round(seconds, 1),
        }
    )

    out_dir = Path("results") if split == "test" else Path("results/dev")
    stem = f"llm_{slugify(cfg['model_name'])}_{len(shots)}shot"
    save_result(result, out_dir / f"{stem}.json")
    save_predictions(
        make_frame(split, evaluation["row_id"], evaluation["label"], preds),
        out_dir / f"preds_{stem}.csv",
    )
    print(f"resultado em {out_dir / (stem + '.json')}")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", dest="model_name", help="modelo do Hub (padrão: o da config)")
    parser.add_argument("--shots-per-class", type=int, dest="shots_per_class")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max-eval", type=int, dest="max_eval", help="subconjunto estratificado")
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--max-new-tokens", type=int, dest="max_new_tokens")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--latency-samples", type=int, dest="latency_samples")
    return parser.parse_args(argv)


def main() -> None:
    options = vars(parse_args())
    result = run(load_config(), options)
    print(
        f"\n{result['split'].upper()}: macro-F1={result['macro_f1']:.4f}  "
        f"acurácia={result['accuracy']:.4f}  respostas inválidas={result['invalid_share']:.1%}  "
        f"latência p50={result['latency']['p50_ms']:.0f} ms"
    )
    print("F1 por classe:", {k: round(v, 4) for k, v in result["f1_per_class"].items()})


if __name__ == "__main__":
    main()
