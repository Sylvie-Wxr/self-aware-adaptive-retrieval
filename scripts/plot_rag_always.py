import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def main():
    parser = argparse.ArgumentParser(description="Plot RAG sweep curves from rag_sweep_metrics.csv")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to rag_sweep_metrics.csv (output by evaluation.py)",
    )
    parser.add_argument(
        "--conf-csv",
        type=str,
        default=None,
        help="Optional path to rag_confusions.csv (output by evaluation.py) for plotting best-config confusion matrix.",
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

    # Sort top-k
    df = df.sort_values(["setting_name", "top_k"])
      
    # Compare with_labels_mesh vs no_labels_mesh  
    compare_settings = ["with_labels_mesh", "no_labels_mesh"]
    df_cmp = df[df["setting_name"].isin(compare_settings)].copy()

    # ====== Accuracy & Hallucination Compare ======
    if not df_cmp.empty and df_cmp["setting_name"].nunique() == 2:
        fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

        # 1) Accuracy vs top_k
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
        ax_acc.set_title("Accuracy vs top_k (with vs without labels+MeSH)")
        ax_acc.legend()
        ax_acc.set_xticks(sorted(df_cmp["top_k"].unique()))  

        # 2) Hallucination vs top_k
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
        ax_hallu.set_title("Hallucination vs top_k (with vs without labels+MeSH)")
        ax_hallu.set_xticks(sorted(df_cmp["top_k"].unique())) 

        plt.tight_layout()
        cmp_path = os.path.join(out_dir, "compare_with_without_labels_mesh.png")
        plt.savefig(cmp_path)
        plt.close()
        print(f"[INFO] Saved {cmp_path}")
    else:
        print("[WARN] Comparison figure not created: could not find both with/without settings in CSV.")
    
    # ====== All curves in one ======
    fig2, ax = plt.subplots(figsize=(8, 5))

    for setting_name, group in df_cmp.groupby("setting_name"):
        group = group.sort_values("top_k")

        # Accuracy
        ax.plot(
            group["top_k"],
            group["accuracy"],
            marker="o",
            linestyle="-",
            label=f"{setting_name} - accuracy",
        )

        # Hallucination
        ax.plot(
            group["top_k"],
            group["hallucination"],
            marker="s",
            linestyle="--",
            label=f"{setting_name} - hallucination",
        )

    ax.set_xlabel("top_k")
    ax.set_ylabel("Rate")
    ax.set_title("Accuracy & Hallucination vs top_k (All curves)")
    ax.set_xticks(sorted(df_cmp["top_k"].unique()))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    all_curves_path = os.path.join(out_dir, "all_curves_in_one.png")
    plt.tight_layout()
    plt.savefig(all_curves_path)
    plt.close()

    print(f"[INFO] Saved {all_curves_path}")
    
    # ====== Confusion metrics based on best config ======
    if args.conf_csv is not None:
        conf_path = args.conf_csv
        if not os.path.exists(conf_path):
            print(f"[WARN] conf-csv file not found: {conf_path}")
        else:
            df_conf = pd.read_csv(conf_path)

            # Best config: smallest hallucination
            # If equal, based on biggest accuracy
            df_best = df.sort_values(
                by=["hallucination", "accuracy"],
                ascending=[True, False],
            )
            best = df_best.iloc[0]

            best_setting = best["setting_name"]
            best_k = int(best["top_k"])
            best_inc_labels = int(best["include_labels"])
            best_inc_meshes = int(best["include_meshes"])

            print("\n[INFO] Best config selected from CSV:")
            print(
                f"  setting={best_setting}, top_k={best_k}, "
                f"include_labels={best_inc_labels}, include_meshes={best_inc_meshes}"
            )
            print(
                f"  accuracy={best['accuracy']:.4f}, hallucination={best['hallucination']:.4f}"
            )

            row = df_conf[
                (df_conf["setting_name"] == best_setting)
                & (df_conf["top_k"] == best_k)
                & (df_conf["include_labels"] == best_inc_labels)
                & (df_conf["include_meshes"] == best_inc_meshes)
            ]

            if row.empty:
                print("[WARN] No matching row found in conf-csv for best config.")
            else:
                row = row.iloc[0]
                # Order: [yy, yn, ym, ny, nn, nm, my, mn, mm]
                cm_vals = [
                    row["cm_yy"], row["cm_yn"], row["cm_ym"],
                    row["cm_ny"], row["cm_nn"], row["cm_nm"],
                    row["cm_my"], row["cm_mn"], row["cm_mm"],
                ]
                cm = np.array(cm_vals).reshape(3, 3)
                labels = ["yes", "no", "maybe"]

                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
                plt.figure()
                disp.plot(values_format="d")
                plt.title(
                    f"Confusion Matrix (best RAG config)\n"
                    f"{best_setting}, top_k={best_k}, "
                    f"acc={best['accuracy']:.3f}, hallu={best['hallucination']:.3f}"
                )
                plt.tight_layout()

                cm_out_path = os.path.join(out_dir, "confusion_matrix_best_rag.png")
                plt.savefig(cm_out_path)
                plt.close()
                print(f"[INFO] Saved {cm_out_path}")
    else:
        print("[INFO] --conf-csv not provided; skip confusion matrix plotting.")

if __name__ == "__main__":
    main()
