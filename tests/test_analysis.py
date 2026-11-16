"""Testes da análise de erros, com previsões pequenas de resultado conhecido."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    _macro_f1,
    accuracy_by_length,
    accuracy_by_rating,
    agreement,
    bootstrap_comparisons,
    common_rows,
    compare_models,
    confusion_normalized,
    export_categories,
    find_predictions,
    has_annotations,
    paired_bootstrap,
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


def test_macro_f1_rapido_bate_com_o_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=500)
    y_pred = np.where(rng.random(500) < 0.7, y_true, rng.integers(0, 3, size=500))
    assert _macro_f1(y_true, y_pred) == pytest.approx(split_metrics(y_true, y_pred)["macro_f1"])


def test_bootstrap_de_modelos_identicos_da_diferenca_zero():
    y = np.random.default_rng(1).integers(0, 3, size=300)
    result = paired_bootstrap(y, y, y, n_boot=50, seed=0)
    assert result["diff"] == 0.0
    assert result["ci95"] == [0.0, 0.0]


def test_bootstrap_de_modelo_perfeito_contra_modelo_errado_exclui_o_zero():
    y = np.random.default_rng(2).integers(0, 3, size=300)
    wrong = (y + 1) % 3
    result = paired_bootstrap(y, y, wrong, n_boot=100, seed=0)
    assert result["diff"] == pytest.approx(1.0)
    assert result["ci95"][0] > 0.9
    assert result["share_a_better"] == 1.0
    # A ordem inverte o sinal.
    assert paired_bootstrap(y, wrong, y, n_boot=100, seed=0)["diff"] == pytest.approx(-1.0)


def test_bootstrap_e_reprodutivel_e_o_intervalo_contem_a_diferenca_observada():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, size=800)
    a = np.where(rng.random(800) < 0.8, y, rng.integers(0, 3, size=800))
    b = np.where(rng.random(800) < 0.7, y, rng.integers(0, 3, size=800))
    first = paired_bootstrap(y, a, b, n_boot=200, seed=5)
    assert first == paired_bootstrap(y, a, b, n_boot=200, seed=5)
    assert first["ci95"][0] < first["diff"] < first["ci95"][1]
    assert first["diff"] > 0


def test_comparacoes_do_bootstrap_escolhem_os_pares_certos():
    base = preds(range(60), [0, 1, 2] * 20, [0, 1, 2] * 20)
    aligned = {"TF-IDF": base, "BERTimbau": base, "Qwen 9-shot": base}
    keys = set(bootstrap_comparisons(aligned, seed=0))
    assert keys == {"BERTimbau - TF-IDF", "Qwen 9-shot - TF-IDF", "BERTimbau - Qwen 9-shot"}


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


def test_amostra_anotada_e_detectada_para_nao_ser_regravada(tmp_path):
    path = tmp_path / "amostra.csv"
    assert has_annotations(path) is False  # nem existe
    pd.DataFrame({"row_id": [1, 2], "categoria": ["", None]}).to_csv(path, index=False)
    assert has_annotations(path) is False  # existe, mas sem anotação
    pd.DataFrame({"row_id": [1, 2], "categoria": ["ironia", None]}).to_csv(path, index=False)
    assert has_annotations(path) is True


def test_exportar_categorias_remove_o_texto_e_mantem_ids_e_categoria(tmp_path):
    annotated = tmp_path / "amostra.csv"
    pd.DataFrame(
        {
            "row_id": [7, 8],
            "rating": [3, 4],
            "verdadeiro": ["neutro", "positivo"],
            "previsto": ["positivo", "neutro"],
            "text": ["texto da avaliação", "outro texto"],
            "categoria": ["rótulo duvidoso", "avaliação mista"],
        }
    ).to_csv(annotated, index=False)

    out = tmp_path / "saida" / "categorias.csv"
    frame = export_categories(annotated, out)

    assert "text" not in frame.columns
    saved = pd.read_csv(out)
    assert saved["row_id"].tolist() == [7, 8]
    assert saved["categoria"].tolist() == ["rótulo duvidoso", "avaliação mista"]
    assert "texto da avaliação" not in out.read_text(encoding="utf-8")


def test_encontra_as_previsoes_pelo_nome(tmp_path):
    for name in (
        "preds_tfidf.csv",
        "preds_bertimbau.csv",
        "preds_llm_qwen-qwen3-4b_9shot.csv",
        "x.csv",
    ):
        (tmp_path / name).write_text("split,row_id,y_true,y_pred\n", encoding="utf-8")
    found = find_predictions(tmp_path)
    assert set(found) == {"TF-IDF", "BERTimbau", "qwen-qwen3-4b 9shot"}  # sem JSON: nome do arquivo


def test_nome_do_llm_vem_do_json_de_resultado(tmp_path):
    for shots, name in ((0, "0shot"), (9, "9shot")):
        (tmp_path / f"preds_llm_qwen-qwen3-4b_{name}.csv").write_text("x\n", encoding="utf-8")
        payload = {"model": "Qwen/Qwen3-4B", "prompt": {"shots": [{}] * shots}}
        (tmp_path / f"llm_qwen-qwen3-4b_{name}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    assert set(find_predictions(tmp_path)) == {"Qwen3-4B zero-shot", "Qwen3-4B 9-shot"}
