"""Testes das previsões do BERT: partes puras e um modelo minúsculo."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.predict_bert import check_against_training, predict_frame, resolve_max_length, softmax
from tests.test_train_bert import corpus, tiny  # noqa: F401  (fixture reaproveitada)


def test_softmax_soma_um_e_e_estavel_com_valores_grandes():
    logits = np.array([[1000.0, 0.0, -1000.0], [1.0, 2.0, 3.0]])
    proba = softmax(logits)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(np.isfinite(proba))
    assert proba[0].argmax() == 0
    assert proba[1].argmax() == 2


def test_max_length_prefere_argumento_depois_json_depois_recalculo(tmp_path):
    saved = tmp_path / "bertimbau.json"
    saved.write_text(json.dumps({"config": {"max_length": 96}}), encoding="utf-8")

    class Tokenizer:
        def __call__(self, texts, truncation):
            return {"input_ids": [[0] * 50 for _ in texts]}

    texts = ["a", "b", "c"]
    assert resolve_max_length(64, saved, Tokenizer(), texts, 95) == (64, "argumento")
    assert resolve_max_length(None, saved, Tokenizer(), texts, 95) == (96, str(saved))
    value, origin = resolve_max_length(None, tmp_path / "nao_existe.json", Tokenizer(), texts, 95)
    assert value == 56  # 50 tokens arredondados para múltiplo de 8
    assert origin == "recalculado do treino"


def make_predictions(y_true, y_pred) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": "test",
            "row_id": range(len(y_true)),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )


def test_conferencia_com_o_treino(tmp_path):
    y_true = [0, 1, 2, 2, 1, 0]
    perfect = make_predictions(y_true, y_true)
    saved = tmp_path / "bertimbau.json"
    saved.write_text(json.dumps({"test": {"macro_f1": 1.0}}), encoding="utf-8")

    ok = check_against_training(perfect, saved)
    assert ok["confere"] is True

    wrong = make_predictions(y_true, [0, 0, 0, 0, 0, 0])
    assert check_against_training(wrong, saved)["confere"] is False
    assert check_against_training(perfect, tmp_path / "nao_existe.json") is None


def test_predict_frame_devolve_classe_e_probabilidades_na_ordem_original(tiny):  # noqa: F811
    import torch

    tokenizer, config, model_cls = tiny
    torch.manual_seed(0)
    model = model_cls(config)
    texts, labels = corpus(30, seed=4)
    frame = pd.DataFrame({"row_id": np.arange(100, 130), "text": texts, "label": labels})

    result = predict_frame(model, tokenizer, frame, "test", max_length=16, batch_size=7)

    assert len(result) == 30
    assert result["row_id"].tolist() == list(range(100, 130))
    assert result["y_true"].tolist() == labels
    proba = result[["p_negativo", "p_neutro", "p_positivo"]].to_numpy()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert (proba.argmax(axis=1) == result["y_pred"].to_numpy()).all()
    with pytest.raises(KeyError):
        result["texto"]  # o texto nunca vai para as previsões
