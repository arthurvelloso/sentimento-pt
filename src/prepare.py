"""Prepara os dados: baixa, rotula, limpa, separa e grava o manifesto do split.

Rode com ``make data``. Gera ``splits/manifest.csv`` e ``splits/meta.json``,
que são versionados. Os textos ficam só em ``data/`` (fora do git).
"""

from __future__ import annotations

import json

import pandas as pd

from src.config import load_config
from src.data import (
    LABELS,
    clean,
    download_raw,
    make_splits,
    save_manifest,
    sha256_of,
)


def main() -> None:
    config = load_config()
    data = config["data"]

    raw_path = download_raw(config)
    if not data.get("sha256"):
        print(
            f"\nSHA-256 do CSV: {sha256_of(raw_path)}\n"
            "Cole em `data.sha256` no configs/default.yaml para fixar o arquivo.\n"
        )

    raw = pd.read_csv(raw_path, usecols=[data["text_col"], data["rating_col"]])
    clean_frame, stats = clean(raw, data["text_col"], data["rating_col"])
    split = make_splits(clean_frame, config["split"], int(config["seed"]))
    save_manifest(clean_frame, split, config, stats)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    for name in ("train", "val", "test"):
        part = clean_frame[split == name]
        shares = part["label"].value_counts(normalize=True).sort_index()
        resumo = ", ".join(f"{LABELS[int(k)]} {100 * v:.1f}%" for k, v in shares.items())
        print(f"{name:5s} n={len(part):6d} | {resumo}")
    print(f"\nmanifesto em {config['split']['manifest_path']}")


if __name__ == "__main__":
    main()
