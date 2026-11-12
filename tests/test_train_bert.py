"""Testes do fine-tuning.

Os testes puros rodam em qualquer ambiente (inclusive na CI leve). Os que treinam
usam um BERT minúsculo com pesos aleatórios e um tokenizador montado na hora:
validam toda a fiação do Trainer sem baixar o BERTimbau. Sem torch, são pulados.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.train_bert import (
    TokenizedDataset,
    compute_max_length,
    compute_metrics,
    resolve_results_path,
    stratified_subset,
)

# ------------------------------------------------------------------ puros


def test_max_length_cobre_o_percentil_e_arredonda_para_multiplo_de_8():
    lengths = list(range(1, 101))  # p95 = 95.05 -> 96 é o múltiplo de 8 seguinte
    assert compute_max_length(lengths, 95) == 96
    assert compute_max_length(lengths, 50) == 56  # p50 = 50.5 -> 51 -> 56


def test_max_length_respeita_minimo_e_maximo():
    assert compute_max_length([3, 4, 5], 95) == 16  # piso
    assert compute_max_length([2000] * 10, 95) == 512  # teto do BERT


def test_max_length_sem_dados_da_erro():
    with pytest.raises(ValueError, match="sem comprimentos"):
        compute_max_length([], 95)


def test_compute_metrics_calcula_macro_f1_e_acuracia():
    logits = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5], [5, 0, 0]], dtype=float)
    labels = np.array([0, 1, 2, 1])  # o último erra
    metrics = compute_metrics(SimpleNamespace(predictions=logits, label_ids=labels))
    assert metrics["accuracy"] == pytest.approx(0.75)
    # F1 por classe: 0 -> 2/3, 1 -> 2/3, 2 -> 1.0
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)


def test_compute_metrics_aceita_predicoes_em_tupla():
    logits = np.array([[1.0, 0.0], [0.0, 1.0]])
    pred = SimpleNamespace(predictions=(logits, np.zeros(2)), label_ids=np.array([0, 1]))
    assert compute_metrics(pred)["accuracy"] == 1.0


def make_frame(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    labels = rng.choice([0, 1, 2], size=n, p=[0.26, 0.12, 0.62])
    return pd.DataFrame({"text": [f"t{i}" for i in range(n)], "label": labels})


def test_subconjunto_estratificado_mantem_proporcao_e_e_reprodutivel():
    frame = make_frame(5000)
    first = stratified_subset(frame, 500, seed=1)
    assert len(first) == 500
    overall = frame["label"].value_counts(normalize=True).sort_index().to_numpy()
    share = first["label"].value_counts(normalize=True).sort_index().to_numpy()
    assert np.allclose(share, overall, atol=0.03)
    assert first.equals(stratified_subset(frame, 500, seed=1))


def test_subconjunto_none_ou_maior_que_o_total_devolve_tudo():
    frame = make_frame(100)
    assert stratified_subset(frame, None, seed=0) is frame
    assert stratified_subset(frame, 500, seed=0) is frame


def test_dataset_tokenizado_devolve_um_item_por_indice():
    encodings = {
        "input_ids": [[1, 5, 2], [1, 6, 7, 2]],
        "attention_mask": [[1, 1, 1], [1, 1, 1, 1]],
    }
    dataset = TokenizedDataset(encodings, [0, 2])
    assert len(dataset) == 2
    assert dataset[1] == {"input_ids": [1, 6, 7, 2], "attention_mask": [1, 1, 1, 1], "labels": 2}
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        TokenizedDataset(encodings, [0])


def test_seeds_extras_nao_sobrescrevem_o_resultado_principal():
    config = load_config("configs/default.yaml")
    main = resolve_results_path(config, int(config["seed"]), None)
    other = resolve_results_path(config, 7, None)
    assert main.name == "bertimbau.json"
    assert other.name == "bertimbau_seed7.json"
    assert str(resolve_results_path(config, 7, "x/y.json")) == "x/y.json"


# ------------------------------------------------- com modelo minúsculo (torch)

WORDS = {
    0: ["horrivel", "pessimo", "quebrou", "atrasou", "decepcao"],
    1: ["razoavel", "mediano", "normal", "comum", "regular"],
    2: ["otimo", "excelente", "adorei", "perfeito", "recomendo"],
}
FILLER = ["produto", "chegou", "entrega", "loja", "compra", "preco", "embalagem"]
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def corpus(n: int, seed: int) -> tuple[list[str], list[int]]:
    rng = np.random.default_rng(seed)
    labels = rng.choice([0, 1, 2], size=n, p=[0.3, 0.2, 0.5])
    texts = []
    for label in labels:
        picked = list(rng.choice(WORDS[int(label)], size=2)) + list(rng.choice(FILLER, size=3))
        rng.shuffle(picked)
        texts.append(" ".join(picked))
    return texts, [int(x) for x in labels]


@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for word in [w for group in WORDS.values() for w in group] + FILLER:
        vocab[word] = len(vocab)

    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=3,
        pad_token_id=vocab["[PAD]"],
    )
    return tokenizer, config, BertForSequenceClassification


HP = {
    "epochs": 6,
    "learning_rate": 3e-3,
    "batch_size": 16,
    "eval_batch_size": 64,
    "weight_decay": 0.0,
    "warmup_ratio": 0.1,
    "use_cpu": True,
}


def test_fine_tune_aprende_registra_historico_e_fica_com_o_melhor_checkpoint(tiny, tmp_path):
    import torch

    from src.train_bert import fine_tune, make_predict_fn, tokenize

    tokenizer, config, model_cls = tiny
    torch.manual_seed(0)
    model = model_cls(config)

    x_train, y_train = corpus(480, seed=0)
    x_val, y_val = corpus(150, seed=1)
    train_ds = TokenizedDataset(tokenize(tokenizer, x_train, 16), y_train)
    val_ds = TokenizedDataset(tokenize(tokenizer, x_val, 16), y_val)

    trainer, info = fine_tune(
        model, tokenizer, train_ds, val_ds, HP, output_dir=tmp_path / "out", seed=0
    )

    assert len(info["history"]) == HP["epochs"]  # uma avaliação por época
    assert info["best_val_macro_f1"] == max(h["val_macro_f1"] for h in info["history"])
    assert info["best_epoch"] in [h["epoch"] for h in info["history"]]
    assert info["best_val_macro_f1"] > 0.7  # chute seria ~0,33

    predict = make_predict_fn(trainer.model, tokenizer, 16)
    accuracy = float(np.mean(predict(x_val) == np.array(y_val)))
    assert accuracy > 0.7


def test_predict_logits_nao_depende_do_batch_nem_da_ordem(tiny):
    import torch

    from src.train_bert import predict_logits

    tokenizer, config, model_cls = tiny
    torch.manual_seed(0)
    model = model_cls(config)
    texts, _ = corpus(40, seed=3)

    in_batch = predict_logits(model, tokenizer, texts, 16, batch_size=7)
    one_by_one = np.concatenate(
        [predict_logits(model, tokenizer, [t], 16, batch_size=1) for t in texts]
    )
    assert in_batch.shape == (40, 3)
    assert np.allclose(in_batch, one_by_one, atol=1e-4)  # mesma ordem de entrada


def test_predict_logits_lista_vazia(tiny):
    from src.train_bert import predict_logits

    tokenizer, config, model_cls = tiny
    assert predict_logits(model_cls(config), tokenizer, [], 16).shape == (0, 3)
