"""Análise de erros e comparação entre modelos, a partir das previsões por exemplo.

Lê ``results/preds_*.csv`` (TF-IDF, BERTimbau, LLM), junta o texto e a nota original das
avaliações (que ficam fora do git) e produz:

- comparação de todos os modelos no MESMO conjunto de avaliações (o LLM pode ter sido
  avaliado num subconjunto do teste, e comparar números de conjuntos diferentes engana);
- matrizes de confusão normalizadas por linha;
- acurácia por nota original (1 a 5): mostra onde o rótulo "positivo" (notas 4-5) e o
  "neutro" (nota 3) se misturam;
- acurácia por tamanho do texto;
- concordância entre dois modelos (quem acerta o que o outro erra);
- uma amostra de erros para leitura manual, em ``analysis/`` (com o texto, fora do git).

Rode com ``make analysis``. Quando a amostra estiver anotada (coluna ``categoria``),
``python -m src.analysis --categorias analysis/erros_bertimbau_amostra.csv`` conta as categorias.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import load_config
from src.data import LABELS, load_splits
from src.predictions import load_predictions

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
ANALYSIS_DIR = Path("analysis")
LENGTH_BINS = (0, 5, 15, 40, 10_000)
LENGTH_NAMES = ("1-5 palavras", "6-15", "16-40", "mais de 40")
INVALID = -1

# ------------------------------------------------------------ métricas puras


def split_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    """Macro-F1, acurácia e F1 por classe. Previsões inválidas (-1) contam como erro."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = list(range(len(LABELS)))
    per_class = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_per_class": {LABELS[i]: float(per_class[i]) for i in classes},
        "invalid_share": float(np.mean(y_pred == INVALID)),
    }


def confusion_normalized(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    """Matriz de confusão com cada linha somando 1 (fração de cada classe verdadeira).

    Previsões inválidas (-1) ficam de fora da matriz, então a linha pode somar menos de 1.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(LABELS)
    counts = np.zeros((n, n))
    for true, pred in zip(y_true, y_pred, strict=True):
        if 0 <= pred < n:
            counts[true, pred] += 1
    totals = np.bincount(y_true, minlength=n).astype(float)
    return counts / np.where(totals == 0, 1, totals)[:, None]


def common_rows(predictions: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restringe todos os modelos aos exemplos que TODOS avaliaram, na mesma ordem."""
    shared = set.intersection(*(set(frame["row_id"]) for frame in predictions.values()))
    return {
        name: frame[frame["row_id"].isin(shared)].sort_values("row_id").reset_index(drop=True)
        for name, frame in predictions.items()
    }


def compare_models(predictions: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Métricas de cada modelo no conjunto comum de exemplos."""
    aligned = common_rows(predictions)
    rows = []
    for name, frame in aligned.items():
        row = {"model": name} | split_metrics(frame["y_true"], frame["y_pred"])
        rows.append(row)
    return rows


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 direto de contagens (bem mais rápido que o sklearn dentro do bootstrap)."""
    scores = []
    for label in range(len(LABELS)):
        truth, pred = y_true == label, y_pred == label
        tp = np.sum(truth & pred)
        denominator = 2 * tp + np.sum(~truth & pred) + np.sum(truth & ~pred)
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def paired_bootstrap(
    y_true: Sequence[int],
    pred_a: Sequence[int],
    pred_b: Sequence[int],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Diferença de macro-F1 (A menos B) com intervalo de 95% por bootstrap pareado.

    Reamostra as avaliações do teste com reposição, mantendo a mesma amostra para A e B
    (pareado), e recalcula a diferença. O intervalo mede a variação **da amostra de teste**;
    não inclui a variação do treino (seed, inicialização), que exigiria treinar de novo.
    """
    y_true, pred_a, pred_b = (np.asarray(x) for x in (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = _macro_f1(y_true[idx], pred_a[idx]) - _macro_f1(y_true[idx], pred_b[idx])
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": _macro_f1(y_true, pred_a) - _macro_f1(y_true, pred_b),
        "ci95": [float(low), float(high)],
        "share_a_better": float(np.mean(diffs > 0)),
        "n_boot": n_boot,
    }


def bootstrap_comparisons(aligned: dict[str, pd.DataFrame], seed: int) -> dict[str, Any]:
    """Diferenças de macro-F1 contra o TF-IDF e do BERTimbau contra cada LLM."""
    pairs: list[tuple[str, str]] = []
    if "TF-IDF" in aligned:
        pairs += [(name, "TF-IDF") for name in aligned if name != "TF-IDF"]
    if "BERTimbau" in aligned:
        pairs += [("BERTimbau", name) for name in aligned if name not in ("BERTimbau", "TF-IDF")]
    result = {}
    for a, b in pairs:
        result[f"{a} - {b}"] = paired_bootstrap(
            aligned[a]["y_true"], aligned[a]["y_pred"], aligned[b]["y_pred"], seed=seed
        )
    return result


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, int]:
    """Quem acerta o que o outro erra, nos exemplos que ambos avaliaram."""
    merged = a.merge(b, on="row_id", suffixes=("_a", "_b"), validate="one_to_one")
    right_a = merged["y_pred_a"] == merged["y_true_a"]
    right_b = merged["y_pred_b"] == merged["y_true_b"]
    both_wrong = ~right_a & ~right_b
    return {
        "n": int(len(merged)),
        "ambos_acertam": int((right_a & right_b).sum()),
        "so_a_acerta": int((right_a & ~right_b).sum()),
        "so_b_acerta": int((~right_a & right_b).sum()),
        "ambos_erram": int(both_wrong.sum()),
        "ambos_erram_igual": int((both_wrong & (merged["y_pred_a"] == merged["y_pred_b"])).sum()),
    }


def accuracy_by_rating(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Por nota original (1-5): acurácia e para onde vão as previsões.

    Espera as colunas ``rating``, ``y_true`` e ``y_pred``.
    """
    rows = []
    for rating, part in frame.groupby("rating"):
        shares = {LABELS[c]: float((part["y_pred"] == c).mean()) for c in range(len(LABELS))}
        rows.append(
            {
                "rating": int(rating),
                "n": int(len(part)),
                "accuracy": float((part["y_pred"] == part["y_true"]).mean()),
                "previsto_como": shares,
            }
        )
    return rows


def accuracy_by_length(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Acurácia por faixa de tamanho do texto, em palavras.

    Espera as colunas ``text``, ``y_true`` e ``y_pred``.
    """
    words = frame["text"].str.split().str.len()
    bucket = pd.cut(words, bins=LENGTH_BINS, labels=LENGTH_NAMES, include_lowest=True)
    rows = []
    for name in LENGTH_NAMES:
        part = frame[bucket == name]
        if len(part):
            rows.append(
                {
                    "faixa": name,
                    "n": int(len(part)),
                    "accuracy": float((part["y_pred"] == part["y_true"]).mean()),
                }
            )
    return rows


def sample_errors(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Amostra de ``n`` erros, proporcional ao tipo de erro (classe verdadeira -> prevista).

    Todo tipo de erro presente entra com pelo menos um exemplo. As avaliações inválidas
    (``y_pred = -1``) formam o seu próprio tipo.
    """
    errors = frame[frame["y_pred"] != frame["y_true"]]
    if errors.empty:
        return errors.copy()
    n = min(n, len(errors))
    rng = np.random.default_rng(seed)
    groups = {key: part for key, part in errors.groupby(["y_true", "y_pred"])}
    shares = {key: len(part) / len(errors) for key, part in groups.items()}
    quota = {key: max(1, round(share * n)) for key, share in shares.items()}
    while sum(quota.values()) > n:  # o arredondamento pode passar do alvo
        biggest = max(quota, key=lambda key: quota[key])
        quota[biggest] -= 1
    picked = [
        part.iloc[rng.choice(len(part), size=min(quota[key], len(part)), replace=False)]
        for key, part in groups.items()
    ]
    return pd.concat(picked).sample(frac=1, random_state=seed).reset_index(drop=True)


def has_annotations(path: Path) -> bool:
    """Se a amostra de erros já tem categorias preenchidas (para não regravá-la por cima)."""
    return path.exists() and summarize_categories(path)["n_anotados"] > 0


def export_categories(annotated: Path, out: Path) -> pd.DataFrame:
    """Grava a classificação sem o texto das avaliações (só ids, notas e categoria).

    O texto fica de fora porque a base é CC BY-NC-SA 4.0; os ids permitem reconstruí-lo.
    """
    frame = pd.read_csv(annotated).drop(columns=["text"], errors="ignore")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


def summarize_categories(path: Path) -> dict[str, Any]:
    """Conta as categorias anotadas à mão na amostra de erros."""
    frame = pd.read_csv(path)
    tagged = frame["categoria"].dropna().astype(str).str.strip()
    tagged = tagged[tagged != ""]
    counts = Counter(tagged)
    return {"n_amostra": int(len(frame)), "n_anotados": int(len(tagged)), "contagem": dict(counts)}


# ------------------------------------------------------------------ contexto


def build_context(config: dict[str, Any], split: str) -> pd.DataFrame:
    """Texto e nota original de cada avaliação de um split, indexados por ``row_id``."""
    data = config["data"]
    raw = pd.read_csv(data["raw_path"], usecols=[data["rating_col"]])
    frame = load_splits(config)[split][["row_id", "text"]].copy()
    frame["rating"] = raw.loc[frame["row_id"], data["rating_col"]].astype(int).to_numpy()
    return frame


def with_context(predictions: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    return predictions.merge(context, on="row_id", validate="one_to_one")


# ------------------------------------------------------------------- figuras


def plot_confusions(matrices: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(
        1, len(matrices), figsize=(4.6 * len(matrices), 4.2), squeeze=False, layout="constrained"
    )
    for ax, (name, matrix) in zip(axes[0], matrices.items(), strict=True):
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=30)
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("previsto")
        ax.set_ylabel("verdadeiro")
        ax.set_title(name)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                color = "white" if matrix[i, j] > 0.5 else "black"
                ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", color=color)
    fig.colorbar(image, ax=axes[0].tolist(), fraction=0.03)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rating_breakdown(by_model: dict[str, list[dict[str, Any]]], path: Path) -> None:
    """Acurácia por nota original, uma linha por modelo."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, rows in by_model.items():
        ax.plot([r["rating"] for r in rows], [r["accuracy"] for r in rows], marker="o", label=name)
    ax.set_xlabel("nota original da avaliação")
    ax.set_ylabel("acurácia")
    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, 1.02)
    ax.set_title("Acurácia por nota (a nota 3 é a classe neutra)")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(curves: dict[str, list[tuple[int, float]]], path: Path) -> None:
    """Macro-F1 no teste contra o número de exemplos de treino (eixo em escala log)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, points in curves.items():
        xs, ys = zip(*sorted(points), strict=True)
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("exemplos de treino (escala log)")
    ax.set_ylabel("macro-F1 no teste")
    ax.set_title("Quanto dado cada modelo precisa")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------- CLI


def find_predictions(results_dir: Path) -> dict[str, Path]:
    """Arquivos de previsões disponíveis, com um nome curto para cada modelo."""
    names = {"preds_tfidf.csv": "TF-IDF", "preds_bertimbau.csv": "BERTimbau"}
    found = {names[p.name]: p for p in results_dir.glob("preds_*.csv") if p.name in names}
    for path in sorted(results_dir.glob("preds_llm_*.csv")):
        found[llm_label(path)] = path
    return found


def llm_label(preds_path: Path) -> str:
    """Nome legível de um LLM (ex.: "Qwen3-4B 9-shot"), lido do JSON de resultado ao lado.

    Sem o JSON, cai num nome derivado do arquivo.
    """
    stem = preds_path.stem.removeprefix("preds_")  # llm_qwen-qwen3-4b_9shot
    result_path = preds_path.with_name(f"{stem}.json")
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        shots = len(saved.get("prompt", {}).get("shots", []))
        model = str(saved.get("model", stem)).split("/")[-1]
        return f"{model} {shots}-shot" if shots else f"{model} zero-shot"
    return stem.removeprefix("llm_").replace("_", " ")


def ablation_curves(results_dir: Path) -> dict[str, list[tuple[int, float]]]:
    curves: dict[str, list[tuple[int, float]]] = {}
    sources = {"TF-IDF": results_dir / "ablation_tfidf.json"}
    sources["BERTimbau"] = results_dir / "ablation_bertimbau.json"
    for name, path in sources.items():
        if path.exists():
            rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
            curves[name] = [(int(r["n_train"]), float(r["test"]["macro_f1"])) for r in rows]
    return curves


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--amostra", type=int, default=40, help="erros na amostra para leitura")
    parser.add_argument("--modelo-amostra", default="BERTimbau")
    parser.add_argument("--categorias", help="CSV anotado: só conta as categorias e sai")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.categorias:
        summary = summarize_categories(Path(args.categorias))
        out = RESULTS_DIR / "categorias_erros_bertimbau.csv"
        export_categories(Path(args.categorias), out)
        report_path = RESULTS_DIR / "analysis.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["categorias_erros_bertimbau"] = summary
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"classificação sem texto em {out}")
        return

    config = load_config()
    seed = int(config["seed"])
    available = find_predictions(RESULTS_DIR)
    print("previsões encontradas:", ", ".join(available) or "nenhuma")
    if not available:
        raise SystemExit("rode `make ablation-tfidf` (e as previsões do BERT) antes.")

    predictions = {
        name: load_predictions(path, split=args.split) for name, path in available.items()
    }
    predictions = {name: frame for name, frame in predictions.items() if len(frame)}
    aligned = common_rows(predictions)
    n_common = len(next(iter(aligned.values())))
    print(f"exemplos em comum entre os modelos ({args.split}): {n_common}")

    context = build_context(config, args.split)
    report: dict[str, Any] = {
        "split": args.split,
        "n_comum": n_common,
        "comparacao": compare_models(predictions),
        "por_nota": {},
        "por_tamanho": {},
    }
    matrices, by_rating = {}, {}
    for name, frame in aligned.items():
        enriched = with_context(frame, context)
        matrices[name] = confusion_normalized(frame["y_true"], frame["y_pred"])
        by_rating[name] = accuracy_by_rating(enriched)
        report["por_nota"][name] = by_rating[name]
        report["por_tamanho"][name] = accuracy_by_length(enriched)

    names = list(aligned)
    if "TF-IDF" in aligned and "BERTimbau" in aligned:
        report["concordancia_tfidf_bert"] = agreement(aligned["TF-IDF"], aligned["BERTimbau"])
    report["bootstrap_macro_f1"] = bootstrap_comparisons(aligned, seed)

    plot_confusions(matrices, FIGURES_DIR / "confusao.png")
    plot_rating_breakdown(by_rating, FIGURES_DIR / "acuracia_por_nota.png")
    curves = ablation_curves(RESULTS_DIR)
    if len(curves) >= 1:
        plot_ablation(curves, FIGURES_DIR / "ablacao_dados.png")

    target = args.modelo_amostra if args.modelo_amostra in aligned else names[0]
    ANALYSIS_DIR.mkdir(exist_ok=True)
    sample_path = ANALYSIS_DIR / f"erros_{target.lower()}_amostra.csv"
    if has_annotations(sample_path):
        sample = pd.read_csv(sample_path)
        print(f"amostra já anotada em {sample_path}: mantida (apague o arquivo para gerar outra)")
    else:
        sample = sample_errors(with_context(predictions[target], context), args.amostra, seed)
        sample = sample.assign(
            verdadeiro=sample["y_true"].map(dict(enumerate(LABELS))),
            previsto=sample["y_pred"].map(dict(enumerate(LABELS))),
            categoria="",
        )[["row_id", "rating", "verdadeiro", "previsto", "text", "categoria"]]
        sample.to_csv(sample_path, index=False)

    if has_annotations(sample_path):
        report["categorias_erros_bertimbau"] = summarize_categories(sample_path)

    (RESULTS_DIR / "analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for row in report["comparacao"]:
        print(
            f"{row['model']:28s} macro-F1={row['macro_f1']:.4f}  acurácia={row['accuracy']:.4f}  "
            f"inválidas={row['invalid_share']:.1%}"
        )
    print(f"\namostra de {len(sample)} erros de {target} em {sample_path} (fora do git)")
    print(f"relatório em {RESULTS_DIR / 'analysis.json'}; figuras em {FIGURES_DIR}")


if __name__ == "__main__":
    main()
