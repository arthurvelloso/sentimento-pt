"""Testes de rótulo, limpeza, split e manifesto, com dados sintéticos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data import (
    clean,
    load_splits,
    make_splits,
    normalize_text,
    save_manifest,
    to_labels,
)


def test_regra_do_rotulo():
    labels = to_labels(pd.Series([1, 2, 3, 4, 5]))
    assert list(labels) == [0, 0, 1, 2, 2]


def test_nota_invalida_vira_nan():
    labels = to_labels(pd.Series([0, 6, "x", 3]))
    assert labels.isna().tolist() == [True, True, True, False]


def test_normalize_text_ignora_caixa_acento_e_pontuacao():
    assert normalize_text("Ótimo  produto!!!") == "otimo produto"
    assert normalize_text("otimo produto") == normalize_text("ÓTIMO PRODUTO.")
    assert normalize_text("  ...  ") == ""


def make_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_text": [
                "Ótimo produto!",  # 0: mantida (primeira cópia)
                "otimo produto",  # 1: duplicata da 0 (mesma nota) -> sai
                "Chegou quebrado",  # 2: texto igual à 3, notas diferentes -> as duas saem
                "chegou quebrado.",  # 3
                None,  # 4: nulo
                "  !!!  ",  # 5: só pontuação
                "Adorei, recomendo",  # 6: mantida
                "Mais ou menos",  # 7: mantida (nota 3)
            ],
            "overall_rating": [5, 4, 1, 5, 5, 4, 5, 3],
        }
    )


def test_limpeza_remove_o_que_deve_e_conta_cada_passo():
    cleaned, stats = clean(make_raw(), "review_text", "overall_rating")

    assert cleaned["row_id"].tolist() == [0, 6, 7]
    assert cleaned["label"].tolist() == [2, 2, 1]
    assert stats == {
        "raw": 8,
        "invalid_rating": 0,
        "null_text": 1,
        "empty_text": 1,
        "conflicting_duplicates_removed": 2,
        "duplicates_removed": 1,
        "clean": 3,
    }


def test_limpeza_mantem_texto_original_e_nao_o_normalizado():
    cleaned, _ = clean(make_raw(), "review_text", "overall_rating")
    assert cleaned["text"].tolist()[0] == "Ótimo produto!"


def test_limpeza_descarta_nota_invalida():
    raw = pd.DataFrame({"review_text": ["bom", "ruim"], "overall_rating": [5, 9]})
    cleaned, stats = clean(raw, "review_text", "overall_rating")
    assert stats["invalid_rating"] == 1
    assert len(cleaned) == 1


def make_clean(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.choice([0, 1, 2], size=n, p=[0.26, 0.12, 0.62])
    return pd.DataFrame(
        {"row_id": np.arange(n), "text": [f"t{i}" for i in range(n)], "label": labels}
    )


FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def test_split_tamanhos_e_sem_sobreposicao():
    frame = make_clean()
    split = make_splits(frame, FRACTIONS, seed=1)
    counts = split.value_counts()
    assert counts["train"] == 800 and counts["val"] == 100 and counts["test"] == 100
    assert split.notna().all()


def test_split_estratificado_mantem_proporcao_das_classes():
    frame = make_clean(n=5000)
    split = make_splits(frame, FRACTIONS, seed=1)
    overall = frame["label"].value_counts(normalize=True).sort_index()
    for name in ("train", "val", "test"):
        share = frame.loc[split == name, "label"].value_counts(normalize=True).sort_index()
        assert np.allclose(share.to_numpy(), overall.to_numpy(), atol=0.01)


def test_split_reprodutivel_e_depende_da_seed():
    frame = make_clean()
    assert make_splits(frame, FRACTIONS, 1).equals(make_splits(frame, FRACTIONS, 1))
    assert not make_splits(frame, FRACTIONS, 1).equals(make_splits(frame, FRACTIONS, 2))


@pytest.fixture
def project(tmp_path):
    """Config apontando para um CSV bruto sintético e manifesto numa pasta temporária."""
    rng = np.random.default_rng(0)
    n = 400
    raw = pd.DataFrame(
        {
            "review_text": [f"produto numero {i} chegou {rng.integers(1000)}" for i in range(n)],
            "overall_rating": rng.choice([1, 2, 3, 4, 5], size=n),
            "extra": 1,
        }
    )
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)

    config = load_config("configs/default.yaml")
    config["data"] = config["data"] | {"raw_path": str(raw_path), "sha256": None}
    config["split"] = config["split"] | {
        "manifest_path": str(tmp_path / "manifest.csv"),
        "meta_path": str(tmp_path / "meta.json"),
    }
    return config, raw


def write_manifest(config, raw):
    cleaned, stats = clean(raw, "review_text", "overall_rating")
    split = make_splits(cleaned, config["split"], int(config["seed"]))
    save_manifest(cleaned, split, config, stats)
    return cleaned, split


def test_manifesto_ida_e_volta_reconstroi_os_mesmos_splits(project):
    config, raw = project
    cleaned, split = write_manifest(config, raw)

    loaded = load_splits(config)

    assert sum(len(part) for part in loaded.values()) == len(cleaned)
    ids = {name: set(part["row_id"]) for name, part in loaded.items()}
    assert ids["train"].isdisjoint(ids["val"]) and ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    assert ids["test"] == set(cleaned.loc[split == "test", "row_id"])


def test_manifesto_que_nao_bate_com_os_dados_da_erro(project):
    config, raw = project
    write_manifest(config, raw)
    manifest = pd.read_csv(config["split"]["manifest_path"]).iloc[:-5]  # perde 5 linhas
    manifest.to_csv(config["split"]["manifest_path"], index=False)
    with pytest.raises(ValueError, match="manifesto não bate"):
        load_splits(config)


def test_sha256_errado_bloqueia_o_uso_do_arquivo(project):
    config, raw = project
    write_manifest(config, raw)
    config["data"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        load_splits(config)


def test_config_rejeita_fracoes_que_nao_somam_um(tmp_path):
    bad = tmp_path / "c.yaml"
    bad.write_text(
        "seed: 1\ndata: {}\ntfidf: {}\nsplit: {train: 0.5, val: 0.1, test: 0.1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="somar 1"):
        load_config(bad)
