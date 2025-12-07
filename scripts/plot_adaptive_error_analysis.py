import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


LABELS = ["yes", "no", "maybe"]


def normalize_label_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .replace({"y": "yes", "n": "no", "unknown": "maybe"})
    )


def compute_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def compute_hallucination_rate(y_true, y_pred) -> float:
    """
    Overclaim: truth == maybe and pred in {yes, no}
    Contradiction: truth in {yes, no} and pred in {yes, no} and pred != truth
    Safe miss: truth in {yes, no} and pred == maybe  -> Not hallucination
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0

    hallu = 0
    for t, p in zip(y_true, y_pred):
        if t == "maybe" and p in {"yes", "no"}:
            hallu += 1
        elif t in {"yes", "no"} and p in {"yes", "no"} and p != t:
            hallu += 1
    return hallu / len(y_true)


def select_config_from_metrics(metrics_csv: str):
    """
    - Ascending order for hallucination
    - Descending order for accuracy 
    """
    dfm = pd.read_csv(metrics_csv)
    dfm_sorted = dfm.sort_values(
        by=["hallucination", "accuracy"],
        ascending=[True, False],
    )
    best = dfm_sorted.iloc[0]

    return dict(
        setting_name=best["setting_name"],
        top_k=int(best["top_k"]),
        include_labels=int(best["include_labels"]),
        include_meshes=int(best["include_meshes"]),
        accuracy=float(best["accuracy"]),
        hallucination=float(best["hallucination"]),
    )


def filter_decisions_for_config(df_dec: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sub = df_dec[
        (df_dec["setting_name"] == cfg["setting_name"])
        & (df_dec["top_k"] == cfg["top_k"])
        & (df_dec["include_labels"] == cfg["include_labels"])
        & (df_dec["include_meshes"] == cfg["include_meshes"])
    ].copy()
    return sub


def plot_self_vs_retrieval(df_sub: pd.DataFrame, out_dir: str, tag: str = ""):
    """
    1. self-only vs retrieval-only  accuracy / hallucination
    """
    df_sub["gold"] = normalize_label_series(df_sub["gold"])
    df_sub["pred"] = normalize_label_series(df_sub["pred"])

    df_self = df_sub[df_sub["used_retrieval"] == 0]
    df_ret = df_sub[df_sub["used_retrieval"] == 1]

    y_true_self = df_self["gold"].tolist()
    y_pred_self = df_self["pred"].tolist()
    y_true_ret = df_ret["gold"].tolist()
    y_pred_ret = df_ret["pred"].tolist()

    acc_self = compute_accuracy(y_true_self, y_pred_self)
    hallu_self = compute_hallucination_rate(y_true_self, y_pred_self)

    acc_ret = compute_accuracy(y_true_ret, y_pred_ret)
    hallu_ret = compute_hallucination_rate(y_true_ret, y_pred_ret)

    print("[Self vs Retrieval]")
    print(f"  self-only  : n={len(df_self)}, acc={acc_self:.4f}, hallu={hallu_self:.4f}")
    print(f"  retrieval  : n={len(df_ret)}, acc={acc_ret:.4f}, hallu={hallu_ret:.4f}")

    groups = ["self-only", "retrieval"]
    acc_vals = [acc_self, acc_ret]
    hallu_vals = [hallu_self, hallu_ret]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, acc_vals, width, label="Accuracy")
    ax.bar(x + width / 2, hallu_vals, width, label="Hallucination")

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Self-only vs Retrieval-only: Accuracy & Hallucination")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    out_path = os.path.join(out_dir, f"adaptive_self_vs_retrieval_{tag}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def plot_confidence_breakdown(df_sub: pd.DataFrame, out_dir: str, tag: str = ""):
    """
    2: confidence = high/medium/low accuracy & hallucination
    """
    df_sub["gold"] = normalize_label_series(df_sub["gold"])
    df_sub["pred"] = normalize_label_series(df_sub["pred"])
    df_sub["confidence"] = (
        df_sub["confidence"].astype(str).str.strip().str.lower()
    )

    conf_levels = ["high", "medium", "low"]
    acc_vals = []
    hallu_vals = []
    counts = []

    for conf in conf_levels:
        sub = df_sub[df_sub["confidence"] == conf]
        y_true = sub["gold"].tolist()
        y_pred = sub["pred"].tolist()
        acc = compute_accuracy(y_true, y_pred)
        hallu = compute_hallucination_rate(y_true, y_pred)
        acc_vals.append(acc)
        hallu_vals.append(hallu)
        counts.append(len(sub))

        print(
            f"[Confidence {conf}] n={len(sub)}, acc={acc:.4f}, hallu={hallu:.4f}"
        )

    x = np.arange(len(conf_levels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, acc_vals, width, label="Accuracy")
    ax.bar(x + width / 2, hallu_vals, width, label="Hallucination")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n(n={n})" for c, n in zip(conf_levels, counts)]
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Accuracy & Hallucination by Confidence Level")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    out_path = os.path.join(out_dir, f"adaptive_confidence_breakdown_{tag}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def plot_pred_distribution(df_sub: pd.DataFrame, out_dir: str, tag: str = ""):
    """
    3: self-only vs retrieval-only prediction (yes/no/maybe)
    """
    df_sub["pred"] = normalize_label_series(df_sub["pred"])

    df_self = df_sub[df_sub["used_retrieval"] == 0]
    df_ret = df_sub[df_sub["used_retrieval"] == 1]

    def count_labels(series):
        series = series.tolist()
        return [series.count(l) for l in LABELS]

    counts_self = count_labels(df_self["pred"])
    counts_ret = count_labels(df_ret["pred"])

    print("[Pred distribution]")
    print("  Labels:", LABELS)
    print("  self-only  :", counts_self)
    print("  retrieval  :", counts_ret)

    x = np.arange(len(LABELS))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    axes[0].bar(x, counts_self)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(LABELS)
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Self-only predictions (n={len(df_self)})")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(x, counts_ret)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LABELS)
    axes[1].set_title(f"Retrieval predictions (n={len(df_ret)})")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Prediction Distribution: Self-only vs Retrieval-only")
    out_path = os.path.join(out_dir, f"adaptive_pred_distribution_{tag}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Error analysis plots for Adaptive RAG from adaptive_decisions.csv"
    )
    parser.add_argument(
        "--decisions-csv",
        type=str,
        required=True,
        help="Path to adaptive_decisions.csv",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="Optional: path to adaptive_sweep_metrics.csv to auto-pick best config "
             "(min hallucination, then max accuracy).",
    )
    parser.add_argument(
        "--setting-name",
        type=str,
        default=None,
        help="If provided, filter to this setting_name (e.g., 'with_labels_mesh').",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="If provided, filter to this top_k.",
    )
    parser.add_argument(
        "--include-labels",
        type=int,
        default=None,
        help="If provided, filter to this include_labels (0/1).",
    )
    parser.add_argument(
        "--include-meshes",
        type=int,
        default=None,
        help="If provided, filter to this include_meshes (0/1).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save plots. Default: same dir as decisions-csv.",
    )
    args = parser.parse_args()

    df_dec = pd.read_csv(args.decisions_csv)

    # out_dir
    if args.out_dir is None:
        out_dir = os.path.dirname(args.decisions_csv) or "."
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Select configuration: 
    # If command-line arguments are provided, use them;
    # otherwise, automatically choose the best config based on metrics_csv.
    cfg = {}
    if (
        args.setting_name is not None
        and args.top_k is not None
        and args.include_labels is not None
        and args.include_meshes is not None
    ):
        cfg = dict(
            setting_name=args.setting_name,
            top_k=args.top_k,
            include_labels=args.include_labels,
            include_meshes=args.include_meshes,
        )
        print("[INFO] Using config from CLI args:")
        print(cfg)
    elif args.metrics_csv is not None:
        cfg = select_config_from_metrics(args.metrics_csv)
        print("[INFO] Auto-selected best config from metrics CSV:")
        print(cfg)
    else:
        raise ValueError(
            "You must either provide "
            "--setting-name/--top-k/--include-labels/--include-meshes "
            "or provide --metrics-csv for auto-selection."
        )

    df_sub = filter_decisions_for_config(df_dec, cfg)
    if df_sub.empty:
        raise RuntimeError(
            "No rows found in decisions CSV for the chosen config: "
            f"{cfg}"
        )

    tag = f"{cfg['setting_name']}_k{cfg['top_k']}_L{cfg['include_labels']}_M{cfg['include_meshes']}"

    # 3 plots
    plot_self_vs_retrieval(df_sub, out_dir, tag)
    plot_confidence_breakdown(df_sub, out_dir, tag)
    plot_pred_distribution(df_sub, out_dir, tag)


if __name__ == "__main__":
    main()
