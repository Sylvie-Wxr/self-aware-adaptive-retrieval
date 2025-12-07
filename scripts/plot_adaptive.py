import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Plot Adaptive RAG curves from adaptive_metrics.csv"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to adaptive_metrics.csv",
    )
    parser.add_argument(
        "--decisions-csv",
        type=str,
        default=None,
        help="Path to adaptive_decisions.csv (for confusion matrices).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save plots. Default: same dir as csv.",
    )
    args = parser.parse_args()

    csv_path = args.csv
    df = pd.read_csv(csv_path)

    if args.out_dir is None:
        out_dir = os.path.dirname(csv_path) or "."
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # sort for topk
    df = df.sort_values(["setting_name", "top_k"])

    # Compare with_labels_mesh vs no_labels_mesh 
    compare_settings = ["with_labels_mesh", "no_labels_mesh"]
    df_cmp = df[df["setting_name"].isin(compare_settings)].copy()

    # ====== 1) Accuracy & Hallucination vs top_k ======
    if not df_cmp.empty and df_cmp["setting_name"].nunique() == 2:
        fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

        # Accuracy
        ax_acc = axes[0]
        for setting_name, group in df_cmp.groupby("setting_name"):
            group = group.sort_values("top_k")
            ax_acc.plot(
                group["top_k"],
                group["accuracy"],
                marker="o",
                label=setting_name,
            )
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title("Adaptive RAG: Accuracy vs top_k")
        ax_acc.legend()
        ax_acc.set_xticks(sorted(df_cmp["top_k"].unique()))

        # Hallucination
        ax_hallu = axes[1]
        for setting_name, group in df_cmp.groupby("setting_name"):
            group = group.sort_values("top_k")
            ax_hallu.plot(
                group["top_k"],
                group["hallucination"],
                marker="o",
                label=setting_name,
            )
        ax_hallu.set_xlabel("top_k")
        ax_hallu.set_ylabel("Hallucination rate")
        ax_hallu.set_title("Adaptive RAG: Hallucination vs top_k")
        ax_hallu.set_xticks(sorted(df_cmp["top_k"].unique()))

        plt.tight_layout()
        acc_hallu_path = os.path.join(out_dir, "adaptive_acc_hallu_vs_topk.png")
        plt.savefig(acc_hallu_path)
        plt.close()
        print(f"[INFO] Saved {acc_hallu_path}")
    else:
        print("[WARN] Not enough settings for accuracy/hallucination comparison.")

    # ====== 2) used_retrieval_rate vs top_k ======
    if "used_retrieval_rate" in df_cmp.columns:
        plt.figure(figsize=(7, 4))
        for setting_name, group in df_cmp.groupby("setting_name"):
            group = group.sort_values("top_k")
            plt.plot(
                group["top_k"],
                group["used_retrieval_rate"],
                marker="o",
                label=setting_name,
            )
        plt.xlabel("top_k")
        plt.ylabel("Used retrieval rate")
        plt.title("Adaptive RAG: Used-retrieval rate vs top_k")
        plt.xticks(sorted(df_cmp["top_k"].unique()))
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()

        used_ret_path = os.path.join(out_dir, "adaptive_used_retrieval_vs_topk.png")
        plt.tight_layout()
        plt.savefig(used_ret_path)
        plt.close()
        print(f"[INFO] Saved {used_ret_path}")
    else:
        print("[WARN] Column 'used_retrieval_rate' not found in CSV.")

    # ====== 3) Latency vs top_k (self, retrieve, total) ======
    needed_cols = {"avg_t_self_ms", "avg_t_retrieve_ms", "avg_t_total_ms"}
    if needed_cols.issubset(df_cmp.columns):
        fig3, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

        latency_cols = [
            ("avg_t_self_ms", "Self-assess latency (ms)"),
            ("avg_t_retrieve_ms", "Retrieve latency (ms)"),
            ("avg_t_total_ms", "End-to-end latency (ms)"),
        ]

        for ax, (col, ylabel) in zip(axes, latency_cols):
            for setting_name, group in df_cmp.groupby("setting_name"):
                group = group.sort_values("top_k")
                ax.plot(
                    group["top_k"],
                    group[col],
                    marker="o",
                    label=setting_name,
                )
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.3)

        axes[-1].set_xlabel("top_k")
        axes[0].set_title("Adaptive RAG: Latency vs top_k")
        axes[-1].set_xticks(sorted(df_cmp["top_k"].unique()))
        axes[0].legend()

        latency_path = os.path.join(out_dir, "adaptive_latency_vs_topk.png")
        plt.tight_layout()
        plt.savefig(latency_path)
        plt.close()
        print(f"[INFO] Saved {latency_path}")
    else:
        print("[WARN] Some latency columns are missing; skip latency figure.")

    # ====== 4) Confusion matrices for best configs ======
    if args.decisions_csv is not None and os.path.exists(args.decisions_csv):
        df_dec = pd.read_csv(args.decisions_csv)

        # 4.1 Choose 2 best configs
        # a) hallucination lowest（choose higher accuracy when tie）
        df_best_hallu = df.sort_values(
            by=["hallucination", "accuracy"],
            ascending=[True, False],
        )
        best_h = df_best_hallu.iloc[0]

        # b) accuracy highest（choose lower hallucination when tie）
        df_best_acc = df.sort_values(
            by=["accuracy", "hallucination"],
            ascending=[False, True],
        )
        best_a = df_best_acc.iloc[0]

        def plot_conf(best_row, tag: str):
            setting = best_row["setting_name"]
            k = int(best_row["top_k"])
            inc_labels = int(best_row["include_labels"])
            inc_meshes = int(best_row["include_meshes"])

            sub = df_dec[
                (df_dec["setting_name"] == setting)
                & (df_dec["top_k"] == k)
                & (df_dec["include_labels"] == inc_labels)
                & (df_dec["include_meshes"] == inc_meshes)
            ].copy()

            if sub.empty:
                print(f"[WARN] No matching rows in decisions CSV for {tag}")
                return

            # gold / pred are already normalized yes/no/maybe
            y_true = sub["gold"].astype(str).str.strip().str.lower().tolist()
            y_pred = sub["pred"].astype(str).str.strip().str.lower().tolist()

            labels = ["yes", "no", "maybe"]
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            plt.figure()
            disp.plot(values_format="d", cmap=plt.cm.Blues)
            plt.title(
                f"Adaptive RAG Confusion Matrix ({tag})\n"
                f"{setting}, top_k={k}, "
                f"acc={best_row['accuracy']:.3f}, hallu={best_row['hallucination']:.3f}"
            )
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"adaptive_confusion_{tag}.png")
            plt.savefig(out_path)
            plt.close()
            print(f"[INFO] Saved {out_path}")

        # 2 confusion matrix plots
        print("\n[INFO] Best (lowest hallucination) adaptive config:")
        print(best_h)
        plot_conf(best_h, "best_hallucination")

        print("\n[INFO] Best (highest accuracy) adaptive config:")
        print(best_a)
        plot_conf(best_a, "best_accuracy")

    else:
        print("[INFO] decisions CSV not provided or not found; skip confusion matrices.")


if __name__ == "__main__":
    main()
