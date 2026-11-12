"""Dados: download, rótulo, limpeza e split estratificado.

Fluxo (``python -m src.prepare``):
1. Baixar o B2W-Reviews01 (CSV fixado por commit, verificado por SHA-256).
2. Rótulo a partir da nota: 1-2 negativo, 3 neutro, 4-5 positivo.
3. Limpeza ANTES do split: texto vazio, duplicatas e quase-duplicatas.
   Fazer isso depois deixaria a mesma avaliação cair em treino e teste, o que
   infla todas as métricas.
4. Split estratificado em treino, validação e teste, com seed fixa.

O que vai para o git é só o manifesto do split (ids de linha do CSV original),
não os textos: o dataset tem licença CC BY-NC-SA 4.0 e fica fora do repositório.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS: tuple[str, ...] = ("negativo", "neutro", "positivo")
RATING_TO_LABEL = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
SPLITS: tuple[str, ...] = ("train", "val", "test")


# --------------------------------------------------------------------- download


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_raw(config: dict[str, Any]) -> Path:
    """Baixa o CSV uma vez e confere o SHA-256, se ele estiver fixado na config."""
    data = config["data"]
    path = Path(data["raw_path"])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"baixando {data['url']}")
        urllib.request.urlretrieve(data["url"], path)  # noqa: S310 (URL fixa, de commit)
    expected = data.get("sha256")
    if expected:
        actual = sha256_of(path)
        if actual != expected:
            raise ValueError(
                f"SHA-256 de {path} não confere.\n  esperado: {expected}\n  obtido:   {actual}\n"
                "O arquivo mudou ou foi corrompido; apague-o e rode de novo."
            )
    return path


# ----------------------------------------------------------------------- rótulo


def to_labels(ratings: pd.Series) -> pd.Series:
    """Nota (1-5) -> classe (0, 1, 2). Notas inválidas viram NaN."""
    numeric = pd.to_numeric(ratings, errors="coerce")
    return numeric.map(RATING_TO_LABEL)


# ---------------------------------------------------------------------- limpeza


def normalize_text(text: str) -> str:
    """Minúsculas, sem acentos, sem pontuação e com espaços colapsados.

    Serve só para achar duplicatas: "Ótimo produto!!" e "otimo produto" ficam iguais.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", without_accents).strip()


def clean(
    frame: pd.DataFrame, text_col: str, rating_col: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aplica rótulo e limpeza, devolvendo o DataFrame limpo e as contagens de cada passo.

    O resultado tem as colunas ``row_id`` (linha no CSV original), ``text`` e ``label``.

    Duplicatas (após normalizar o texto):
    - mesmo texto com rótulos iguais: fica uma cópia (a primeira);
    - mesmo texto com rótulos diferentes: o rótulo é ambíguo e todas as cópias saem.
    Quase-duplicatas "fuzzy" (textos parecidos, mas não iguais) não são tratadas.
    """
    stats: dict[str, Any] = {"raw": int(len(frame))}
    work = pd.DataFrame(
        {
            "row_id": frame.index.to_numpy(),
            "text": frame[text_col].astype("string"),
            "label": to_labels(frame[rating_col]),
        }
    )

    invalid = work["label"].isna()
    stats["invalid_rating"] = int(invalid.sum())
    work = work[~invalid]

    before = len(work)
    work = work[work["text"].notna()].copy()
    stats["null_text"] = int(before - len(work))

    work["norm"] = [normalize_text(str(t)) for t in work["text"]]
    empty = work["norm"] == ""
    stats["empty_text"] = int(empty.sum())  # só espaços, pontuação ou emojis
    work = work[~empty]

    label_count = work.groupby("norm")["label"].transform("nunique")
    conflicting = label_count > 1
    stats["conflicting_duplicates_removed"] = int(conflicting.sum())
    work = work[~conflicting]

    duplicated = work.duplicated(subset="norm", keep="first")
    stats["duplicates_removed"] = int(duplicated.sum())
    work = work[~duplicated]

    out = work[["row_id", "text", "label"]].reset_index(drop=True)
    out["label"] = out["label"].astype(int)
    out["text"] = out["text"].astype(str)
    stats["clean"] = int(len(out))
    return out, stats


# ------------------------------------------------------------------------ split


def make_splits(clean_frame: pd.DataFrame, fractions: dict[str, float], seed: int) -> pd.Series:
    """Split estratificado por rótulo. Devolve o nome do split de cada linha."""
    labels = clean_frame["label"].to_numpy()
    index = np.arange(len(clean_frame))

    rest, test_idx = train_test_split(
        index, test_size=fractions["test"], stratify=labels, random_state=seed
    )
    val_share = fractions["val"] / (fractions["train"] + fractions["val"])
    train_idx, val_idx = train_test_split(
        rest, test_size=val_share, stratify=labels[rest], random_state=seed
    )

    names = np.empty(len(clean_frame), dtype=object)
    names[train_idx], names[val_idx], names[test_idx] = "train", "val", "test"
    return pd.Series(names, index=clean_frame.index, name="split")


def save_manifest(
    clean_frame: pd.DataFrame, split: pd.Series, config: dict[str, Any], stats: dict[str, Any]
) -> None:
    """Grava o manifesto (row_id, split) e um meta.json com contagens e o SHA-256 do CSV."""
    cfg = config["split"]
    manifest_path, meta_path = Path(cfg["manifest_path"]), Path(cfg["meta_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": clean_frame["row_id"], "split": split}).to_csv(
        manifest_path, index=False
    )

    per_split = {}
    for name in SPLITS:
        labels = clean_frame.loc[split == name, "label"]
        counts = labels.value_counts().reindex(range(len(LABELS)), fill_value=0)
        per_split[name] = {
            "n": int(len(labels)),
            "por_classe": {LABELS[i]: int(counts[i]) for i in range(len(LABELS))},
        }
    meta = {
        "seed": config["seed"],
        "source_url": config["data"]["url"],
        "raw_sha256": sha256_of(Path(config["data"]["raw_path"])),
        "rotulo": "1-2 negativo | 3 neutro | 4-5 positivo",
        "limpeza": stats,
        "splits": per_split,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_splits(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Reconstrói treino, validação e teste a partir do CSV original e do manifesto.

    Refaz a limpeza (determinística) e confere que os ids batem com o manifesto
    versionado. Se não baterem, o dado ou a regra de limpeza mudou.
    """
    data = config["data"]
    raw_path = download_raw(config)
    raw = pd.read_csv(raw_path, usecols=[data["text_col"], data["rating_col"]])
    clean_frame, _ = clean(raw, data["text_col"], data["rating_col"])

    manifest = pd.read_csv(config["split"]["manifest_path"])
    if set(manifest["row_id"]) != set(clean_frame["row_id"]):
        raise ValueError(
            "o manifesto não bate com os dados limpos (ids diferentes). "
            "Rode `make data` para regenerar; commite o novo manifesto só se a mudança "
            "for intencional."
        )
    merged = clean_frame.merge(manifest, on="row_id", validate="one_to_one")
    return {
        name: merged[merged["split"] == name].drop(columns="split").reset_index(drop=True)
        for name in SPLITS
    }
