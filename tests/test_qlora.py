"""Testes das partes do QLoRA que não precisam de GPU: exemplos, máscara, plano e LR."""

from __future__ import annotations

import json

import pytest

from src.analysis import llm_label
from src.qlora import (
    IGNORE_INDEX,
    build_example,
    collate,
    learning_rate_at,
    plan_steps,
)


class FakeTokenizer:
    """Um token por caractere; template de chat simples. Basta para testar a máscara."""

    eos_token = "<fim>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and enable_thinking is False
        return "".join(f"[{m['role']}]{m['content']}\n" for m in messages) + "[assistant]"

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text]}


def decode(ids):
    return "".join(chr(i) for i in ids)


def test_exemplo_tem_loss_so_na_resposta_e_termina_no_fim_de_turno():
    example = build_example(FakeTokenizer(), "gostei muito", 2, max_chars=100)
    ids, labels = example["input_ids"], example["labels"]
    assert len(ids) == len(labels)
    supervised = [i for i, y in zip(ids, labels, strict=True) if y != IGNORE_INDEX]
    assert decode(supervised) == "positivo<fim>"
    # Tudo antes da resposta é ignorado, e o prompt é o mesmo da inferência.
    prompt_len = len(ids) - len(supervised)
    assert set(labels[:prompt_len]) == {IGNORE_INDEX}
    assert decode(ids[:prompt_len]).endswith("Avaliação: gostei muito\nSentimento:\n[assistant]")


def test_exemplo_respeita_os_tres_rotulos():
    for label, word in enumerate(("negativo", "neutro", "positivo")):
        labels = build_example(FakeTokenizer(), "x", label, 100)["labels"]
        assert decode([y for y in labels if y != IGNORE_INDEX]) == word + "<fim>"


def test_collate_alinha_com_padding_a_direita_e_ignora_o_padding():
    a = {"input_ids": [1, 2, 3], "labels": [IGNORE_INDEX, 2, 3]}
    b = {"input_ids": [4], "labels": [4]}
    batch = collate([a, b], pad_id=0)
    assert batch["input_ids"] == [[1, 2, 3], [4, 0, 0]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"] == [[IGNORE_INDEX, 2, 3], [4, IGNORE_INDEX, IGNORE_INDEX]]


def test_plano_cobre_cada_exemplo_uma_vez_por_epoca_e_e_reprodutivel():
    steps = plan_steps(n_examples=10, micro_batch=4, grad_accum=2, epochs=1, seed=3)
    flat = [i for step in steps for micro in step for i in micro]
    assert sorted(flat) == list(range(10))
    assert [len(step) for step in steps] == [2, 1]  # sobra um micro-batch menor
    assert steps == plan_steps(10, 4, 2, 1, 3)
    assert steps != plan_steps(10, 4, 2, 1, 4)


def test_plano_com_duas_epocas_embaralha_de_novo():
    steps = plan_steps(n_examples=8, micro_batch=4, grad_accum=1, epochs=2, seed=0)
    first = [i for s in steps[:2] for m in s for i in m]
    second = [i for s in steps[2:] for m in s for i in m]
    assert sorted(first) == sorted(second) == list(range(8))
    assert first != second


def test_learning_rate_sobe_e_depois_cai_a_zero():
    total, base = 100, 2e-4
    rates = [learning_rate_at(s, total, base, 0.1) for s in range(total)]
    assert rates[0] < rates[9] == pytest.approx(base)  # fim do aquecimento (10 passos)
    assert all(a >= b for a, b in zip(rates[9:], rates[10:], strict=False))
    assert rates[-1] == pytest.approx(base / (total - 10))  # último passo, quase zero
    assert learning_rate_at(0, 1, base, 0.03) == pytest.approx(base)  # treino de 1 passo


def test_rotulo_do_llm_com_adaptador_nao_colide_com_o_zero_shot(tmp_path):
    base = {"model": "Qwen/Qwen3-4B", "prompt": {"shots": []}}
    (tmp_path / "llm_qwen-qwen3-4b_0shot.json").write_text(json.dumps(base))
    (tmp_path / "llm_qwen-qwen3-4b-qlora_0shot.json").write_text(
        json.dumps(base | {"adapter": "outputs/qlora/adapter"})
    )
    zero = llm_label(tmp_path / "preds_llm_qwen-qwen3-4b_0shot.csv")
    tuned = llm_label(tmp_path / "preds_llm_qwen-qwen3-4b-qlora_0shot.csv")
    assert zero == "Qwen3-4B zero-shot"
    assert tuned == "Qwen3-4B + QLoRA"
