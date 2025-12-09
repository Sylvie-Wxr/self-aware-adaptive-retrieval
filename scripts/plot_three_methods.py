import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_baseline(path):
    """Reads baseline_metrics_xxx.txt"""
    acc = None
    hallu = None
    latency = None
    with open(path, "r") as f:
        for line in f:
            if line.startswith("accuracy="):
                acc = float(line.split("=")[1])
            if line.startswith("hallucination="):
                hallu = float(line.split("=")[1])
            if line.startswith("avg_t_total_ms="):
                latency = float(line.split("=")[1])
    return acc, hallu, latency


def best_rag(rag_csv):
    df = pd.read_csv(rag_csv)

    # Best hallucination then accuracy
    df_best = df.sort_values(
        by=["hallucination", "accuracy"],
        ascending=[True, False],
    )
    best = df_best.iloc[0]
    return float(best["accuracy"]), float(best["hallucination"]), float(best["avg_t_total_ms"])


def best_adaptive(ad_csv):
    df = pd.read_csv(ad_csv)

    df_best = df.sort_values(
        by=["hallucination", "accuracy"],
        ascending=[True, False],
    )
    best = df_best.iloc[0]
    return float(best["accuracy"]), float(best["hallucination"]), float(best["avg_t_total_ms"])


def plot_three_methods(b_acc, b_hallu, b_lat,
                       r_acc, r_hallu, r_lat,
                       a_acc, a_hallu, a_lat,
                       out_dir):

    methods = ["No-RAG", "RAG-Always", "Adaptive-RAG"]
    accs = [b_acc, r_acc, a_acc]
    hallus = [b_hallu, r_hallu, a_hallu]
    lats = [b_lat, r_lat, a_lat]

    # ------------------------------------
    # 1. Accuracy vs Latency (Scatter Plot)
    # ------------------------------------
    plt.figure(figsize=(7,5))
    plt.scatter(lats, accs, s=150, color=["#1f77b4", "#ff7f0e", "#2ca02c"])

    for m, x, y in zip(methods, lats, accs):
        plt.text(x + 3, y, m, fontsize=10)

    plt.xlabel("Latency (ms)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Latency")
    plt.grid(True, linestyle="--", alpha=0.5)

    out1 = os.path.join(out_dir, "accuracy_vs_latency.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=300)
    plt.close()
    print(f"[INFO] Saved {out1}")

    # ----------------------------------------
    # 2. Hallucination vs Latency (Scatter Plot)
    # ----------------------------------------
    plt.figure(figsize=(7,5))
    plt.scatter(lats, hallus, s=150, color=["#1f77b4", "#ff7f0e", "#2ca02c"])  # 红/橙/绿区分更清楚

    for m, x, y in zip(methods, lats, hallus):
        plt.text(x + 3, y, m, fontsize=10)

    plt.xlabel("Latency (ms)")
    plt.ylabel("Hallucination Rate")
    plt.title("Hallucination vs Latency")
    plt.grid(True, linestyle="--", alpha=0.5)

    out2 = os.path.join(out_dir, "hallucination_vs_latency.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"[INFO] Saved {out2}")

    # -------------------------------------------------
    # 3. Grouped Bar Chart (Accuracy & Hallucination)
    # -------------------------------------------------

    x = np.arange(len(methods))
    width = 0.35

    color_acc = "#1f77b4"
    color_hal = "#ff7f0e" 

    plt.figure(figsize=(8, 6))

    rects1 = plt.bar(x - width/2, accs, width,
                     label='Accuracy', color=color_acc, alpha=0.8)
    rects2 = plt.bar(x + width/2, hallus, width,
                     label='Hallucination Rate', color=color_hal, alpha=0.8)

    # Add label
    plt.bar_label(rects1, padding=3, fmt='%.3f')
    plt.bar_label(rects2, padding=3, fmt='%.3f')

    plt.ylabel('Score / Rate', fontsize=12)
    plt.title('Accuracy vs Hallucination Comparison', fontsize=14)
    plt.xticks(x, methods, fontsize=11)
    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.ylim(0, max(max(accs), max(hallus)) + 0.1)

    out3 = os.path.join(out_dir, "metrics_comparison_bar.png")
    plt.tight_layout()
    plt.savefig(out3, dpi=300)
    plt.close()
    print(f"[INFO] Saved {out3}")
    
     # -------------------------------------------------
    # 4. Accuracy vs Hallucination (Scatter）
    # -------------------------------------------------
    plt.figure(figsize=(7, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue / Orange / Green
    plt.scatter(hallus, accs, s=180, color=colors)

    for m, x, y, c in zip(methods, hallus, accs, colors):
        plt.text(x + 0.003, y, m, fontsize=11, color=c)

    plt.xlabel("Hallucination Rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Hallucination")
    plt.grid(True, linestyle="--", alpha=0.5)

    out4 = os.path.join(out_dir, "acc_vs_hallu_scatter.png")
    plt.tight_layout()
    plt.savefig(out4, dpi=300)
    plt.close()
    print(f"[INFO] Saved {out4}")


def main():
    parser = argparse.ArgumentParser(description="Compare no-rag, RAG-always, and Adaptive-RAG.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--rag", required=True)
    parser.add_argument("--adaptive", required=True)
    parser.add_argument("--out-dir", default=".", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    b_acc, b_hallu, b_lat = read_baseline(args.baseline)
    r_acc, r_hallu, r_lat = best_rag(args.rag)
    a_acc, a_hallu, a_lat = best_adaptive(args.adaptive)

    print("Baseline:", b_acc, b_hallu, b_lat)
    print("RAG-best:", r_acc, r_hallu, r_lat)
    print("Adaptive-best:", a_acc, a_hallu, a_lat)

    plot_three_methods(b_acc, b_hallu, b_lat,
                       r_acc, r_hallu, r_lat,
                       a_acc, a_hallu, a_lat,
                       args.out_dir)


if __name__ == "__main__":
    main()
