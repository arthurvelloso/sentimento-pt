"""Testes da classificação com LLM: prompt, parser, few-shot e geração em lote."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.llm_classify as llm
from src.evaluate import evaluate
from src.llm_classify import (
    INVALID,
    build_messages,
    classify,
    format_review,
    parse_answer,
    render_prompts,
    select_shots,
    slugify,
    truncate,
)

LABELS = ("negativo", "neutro", "positivo")

# ---------------------------------------------------------------------- parser


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("positivo", 2),
        ("Positivo.", 2),
        ("  NEGATIVO\n", 0),
        ("neutro", 1),
        ("Positiva", 2),  # feminino
        ("Negativo, pois o produto quebrou", 0),  # a primeira palavra manda
        ("O sentimento é positivo", 2),  # um único rótulo no meio
        ("negativo ou neutro", 0),  # primeira palavra
        ("é negativo ou neutro", INVALID),  # dois rótulos, nenhum na frente
        ("não é positivo", INVALID),  # negação nunca vira o rótulo
        ("sem sentimento claro", INVALID),
        ("", INVALID),
        ("   ", INVALID),
        ("não sei", INVALID),
        ("bom", INVALID),
    ],
)
def test_parse_answer(answer, expected):
    assert parse_answer(answer) == expected


# ---------------------------------------------------------------------- prompt


def test_truncate_colapsa_espacos_e_corta():
    assert truncate("a   b\n c", 100) == "a b c"
    assert truncate("x" * 50, 10) == "x" * 10 + "…"


def test_format_review_termina_no_gancho_da_resposta():
    prompt = format_review("Chegou  rápido", 100)
    assert prompt == "Avaliação: Chegou rápido\nSentimento:"


def test_build_messages_zero_shot_e_few_shot():
    zero = build_messages("texto", [], 100)
    assert [m["role"] for m in zero] == ["system", "user"]
    assert "negativo, neutro ou positivo" in zero[0]["content"]

    shots = [{"row_id": 1, "text": "ruim", "label": 0}, {"row_id": 2, "text": "ótimo", "label": 2}]
    few = build_messages("texto", shots, 100)
    assert [m["role"] for m in few] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert few[2]["content"] == "negativo"
    assert few[4]["content"] == "positivo"
    assert few[-1]["content"].startswith("Avaliação: texto")


def train_frame(n_per_class: int = 40) -> pd.DataFrame:
    rows = []
    row_id = 0
    for label in range(3):
        for i in range(n_per_class):
            text = f"avaliacao {label} numero {i}" + (" longa" * 100 if i % 4 == 0 else "")
            rows.append({"row_id": row_id, "text": text, "label": label})
            row_id += 1
    return pd.DataFrame(rows)


def test_select_shots_tem_k_por_classe_so_textos_curtos_e_e_reprodutivel():
    train = train_frame()
    shots = select_shots(train, per_class=3, seed=7, max_chars=100)

    assert len(shots) == 9
    assert sorted(s["label"] for s in shots) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert all(len(s["text"]) <= 100 for s in shots)
    assert len({s["row_id"] for s in shots}) == 9  # sem repetição
    assert shots == select_shots(train, per_class=3, seed=7, max_chars=100)
    assert shots != select_shots(train, per_class=3, seed=8, max_chars=100)


def test_select_shots_zero_e_erro_quando_faltam_exemplos():
    assert select_shots(train_frame(), per_class=0, seed=0) == []
    with pytest.raises(ValueError, match="poucos exemplos"):
        select_shots(train_frame(n_per_class=3), per_class=5, seed=0, max_chars=100)


def test_slugify():
    assert slugify("Qwen/Qwen3.5-4B") == "qwen-qwen3-5-4b"


# ------------------------------------------------- inválidas contam como erro


def test_resposta_invalida_conta_como_erro_no_avaliador():
    y_true = [0, 1, 2, 2]
    preds = np.array([0, INVALID, 2, INVALID])
    result = evaluate(
        lambda texts: preds, list("abcd"), y_true, LABELS, name="x", latency_samples=2
    )
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["f1_per_class"]["neutro"] == 0.0
    assert result["macro_f1"] == pytest.approx((1.0 + 0.0 + 2 / 3) / 3)
    assert sum(sum(row) for row in result["confusion_matrix"]["values"]) == 2  # só as válidas


# ----------------------------------------------------------- geração em lote


@pytest.fixture(scope="module")
def tiny_lm():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    words = [
        "negativo",
        "neutro",
        "positivo",
        "avaliação",
        "sentimento",
        "system",
        "user",
        "assistant",
        "produto",
        "chegou",
        "rápido",
        "ruim",
        "ótimo",
        "quebrou",
        ":",
    ]
    vocab = {tok: i for i, tok in enumerate(["[PAD]", "[UNK]", "[EOS]", *words])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=len(vocab),
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_positions=512,
        pad_token_id=vocab["[PAD]"],
        eos_token_id=vocab["[EOS]"],
    )
    return tokenizer, GPT2LMHeadModel(config)


TEXTS = [
    "produto chegou rápido",
    "ruim",
    "ótimo produto",
    "quebrou",
    "produto ruim quebrou",
    "chegou",
    "ótimo",
    "rápido produto ótimo",
]


def test_render_prompts_aplica_o_template_e_ignora_enable_thinking(tiny_lm):
    tokenizer, _ = tiny_lm
    prompts = render_prompts(tokenizer, [build_messages(t, [], 100) for t in TEXTS[:2]])
    assert len(prompts) == 2
    assert prompts[0].endswith("assistant:")
    assert "produto chegou rápido" in prompts[0]


def test_geracao_mantem_a_ordem_e_nao_depende_do_tamanho_do_batch(tiny_lm):
    tokenizer, model = tiny_lm
    prompts = render_prompts(tokenizer, [build_messages(t, [], 100) for t in TEXTS])

    one_by_one = llm.generate_answers(model, tokenizer, prompts, batch_size=1, max_new_tokens=4)
    in_batches = llm.generate_answers(model, tokenizer, prompts, batch_size=3, max_new_tokens=4)

    assert len(one_by_one) == len(TEXTS)
    assert one_by_one == in_batches
    assert llm.generate_answers(model, tokenizer, [], batch_size=4) == []


def test_classify_devolve_classes_validas_ou_invalidas(tiny_lm):
    tokenizer, model = tiny_lm
    preds, answers = classify(
        model, tokenizer, TEXTS, [], max_chars=100, batch_size=4, max_new_tokens=3
    )
    assert preds.shape == (len(TEXTS),)
    assert set(preds.tolist()) <= {INVALID, 0, 1, 2}
    assert len(answers) == len(TEXTS)
    assert preds.tolist() == [parse_answer(a) for a in answers]


def test_classify_conta_as_invalidas_com_respostas_controladas(tiny_lm, monkeypatch):
    tokenizer, model = tiny_lm
    canned = ["positivo", "talvez", "Negativo.", "", "neutro", "não é positivo", "positivo", "x"]
    monkeypatch.setattr(llm, "generate_answers", lambda *a, **k: canned)

    preds, answers = classify(
        model, tokenizer, TEXTS, [], max_chars=100, batch_size=4, max_new_tokens=3
    )

    assert preds.tolist() == [2, INVALID, 0, INVALID, 1, INVALID, 2, INVALID]
    assert answers == canned
    assert float((preds == INVALID).mean()) == 0.5
