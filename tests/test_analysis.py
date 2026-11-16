"""Testes da análise de erros, com previsões pequenas de resultado conhecido."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    accuracy_by_length,
    accuracy_by_rating,
    agreement,
    common_rows,
    compare_models,
    confusion_normalized,
    find_predictions,
    sample_errors,
    split_metrics,
    summarize_categories,
)


def preds(row_ids, y_true, y_pred) -> pd.DataFrame:
    return pd.DataFrame({"split": "test", "row_id": row_ids, "y_true": y_true, "y_pred": y_pred})


def test_split_metrics_valores_conhecidos_e_invalidas_contam_como_erro():
    metrics = split_metrics([0, 1, 2, 2], [0, -1, 2, -1])
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["invalid_share"] == pytest.approx(0.5)
    assert metrics["f1_per_class"]["neutro"] == 0.0
    assert metrics["macro_f1"] == pytest.approx((1.0 + 0.0 + 2 / 3) / 3)


def test_confusao_normalizada_soma_um_por_linha_e_ignora_invalidas():
    matrix = confusion_normalized([0, 0, 1, 1, 2, 2, 2], [0, 1, 1, 1, 2, 0, 2])
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert matrix[0].tolist() == [0.5, 0.5, 0.0]
    assert matrix[2].tolist() == pytest.approx([1 / 3, 0.0, 2 / 3])

    with_invalid = confusion_normalized([0, 0], [0, -1])
    assert with_invalid[0].sum() == pytest.approx(0.5)  # a inválida fica fora da matriz


def test_comparacao_usa_so_os_exemplos_em_comum():
    full = preds([1, 2, 3, 4], [0, 1, 2, 2], [0, 1, 2, 0])
    subset = preds([2, 3], [1, 2], [1, 0])  # o LLM só viu dois exemplos
    aligned = common_rows({"A": full, "B": subset})
    assert aligned["A"]["row_id"].tolist() == [2, 3]
    assert aligned["B"]["row_id"].tolist() == [2, 3]

    table = {row["model"]: row for row in compare_models({"A": full, "B": subset})}
    assert table["A"]["n"] == table["B"]["n"] == 2
    assert table["A"]["accuracy"] == 1.0  # A acertou os dois exemplos comuns
    assert table["B"]["accuracy"] == 0.5


def test_concordancia_entre_dois_modelos():
    a = preds([1, 2, 3, 4], [0, 1, 2, 2], [0, 1, 0, 0])
    b = preds([1, 2, 3, 4], [0, 1, 2, 2], [0, 2, 2, 1])
    result = agreement(a, b)
    assert result == {
        "n": 4,
        "ambos_acertam": 1,  # linha 1
        "so_a_acerta": 1,  # linha 2
        "so_b_acerta": 1,  # linha 3
        "ambos_erram": 1,  # linha 4
        "ambos_erram_igual": 0,  # A previu 0 e B previu 1
    }


def test_acuracia_por_nota_mostra_para_onde_vao_as_previsoes():
    frame = pd.DataFrame(
        {
            "rating": [4, 4, 4, 4, 3, 3],
            "y_true": [2, 2, 2, 2, 1, 1],
            "y_pred": [2, 1, 1, 2, 1, 2],
        }
    )
    by_rating = {row["rating"]: row for row in accuracy_by_rating(frame)}
    assert by_rating[4]["accuracy"] == pytest.approx(0.5)
    assert by_rating[4]["previsto_como"]["neutro"] == pytest.approx(0.5)
    assert by_rating[3]["n"] == 2


def test_acuracia_por_tamanho_agrupa_em_faixas_de_palavras():
    frame = pd.DataFrame(
        {
            "text": ["bom", "muito bom mesmo", " ".join(["palavra"] * 20), " ".join(["x"] * 60)],
            "y_true": [2, 2, 2, 2],
            "y_pred": [2, 2, 0, 2],
        }
    )
    by_length = {row["faixa"]: row for row in accuracy_by_length(frame)}
    assert by_length["1-5 palavras"]["n"] == 2
    assert by_length["16-40"]["accuracy"] == 0.0
    assert by_length["mais de 40"]["accuracy"] == 1.0
    assert "6-15" not in by_length  # faixa vazia não aparece


def test_amostra_de_erros_so_tem_erros_e_cobre_cada_tipo():
    rng = np.random.default_rng(0)
    n = 600
    y_true = rng.choice([0, 1, 2], size=n, p=[0.3, 0.15, 0.55])
    y_pred = np.where(rng.random(n) < 0.3, rng.choice([0, 1, 2], size=n), y_true)
    frame = preds(np.arange(n), y_true, y_pred).assign(text="t", rating=3)

    sample = sample_errors(frame, n=40, seed=1)
    assert len(sample) == 40
    assert (sample["y_pred"] != sample["y_true"]).all()
    assert sample["row_id"].is_unique

    wrong = frame[frame["y_pred"] != frame["y_true"]]
    all_types = set(map(tuple, wrong[["y_true", "y_pred"]].to_numpy().tolist()))
    sampled_types = set(map(tuple, sample[["y_true", "y_pred"]].to_numpy().tolist()))
    assert sampled_types <= all_types
    assert sample.equals(sample_errors(frame, n=40, seed=1))  # reprodutível


def test_amostra_nao_passa_do_total_de_erros():
    frame = preds([1, 2, 3], [0, 1, 2], [0, 2, 2]).assign(text="t", rating=3)  # um erro só
    assert len(sample_errors(frame, n=40, seed=0)) == 1
    perfect = preds([1, 2], [0, 1], [0, 1]).assign(text="t", rating=3)
    assert sample_errors(perfect, n=40, seed=0).empty


def test_resumo_de_categorias_ignora_linhas_sem_anotacao(tmp_path):
    path = tmp_path / "amostra.csv"
    pd.DataFrame(
        {"row_id": [1, 2, 3, 4], "categoria": ["ironia", "ironia", "texto curto", None]}
    ).to_csv(path, index=False)
    summary = summarize_categories(path)
    assert summary == {"n_amostra": 4, "n_anotados": 3, "contagem": {"ironia": 2, "texto curto": 1}}


def test_encontra_as_previsoes_pelo_nome(tmp_path):
    for name in (
        "preds_tfidf.csv",
        "preds_bertimbau.csv",
        "preds_llm_qwen-qwen3-4b_9shot.csv",
        "x.csv",
    ):
        (tmp_path / name).write_text("split,row_id,y_true,y_pred\n", encoding="utf-8")
    found = find_predictions(tmp_path)
    assert set(found) == {"TF-IDF", "BERTimbau", "qwen-qwen3-4b_9shot"}
