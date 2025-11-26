import os
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset

DEFAULT_OUT_DIR = "/home/wu.xinrui/ondemand/dev/self-aware-adaptive-retrieval/results/data_eda"

def main():
    parser = argparse.ArgumentParser(description="Check PubMedQA labeled dataset.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Directory to save EDA outputs.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading PubMedQA datasets...")
    labeled_dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    print(f"labeled_dataset rows: {len(labeled_dataset)}")

    df = pd.DataFrame(labeled_dataset)

    text_report = []

    text_report.append("=== Missing Value Summary ===")
    text_report.append(str(df.isnull().sum()))

    text_report.append("\n=== Dataset Overview ===")
    text_report.append(str(df.describe(include="all")))

    text_report.append("\n=== Label Distribution ===")
    text_report.append(str(df["final_decision"].value_counts()))

    report_path = os.path.join(args.out_dir, "data_overview.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(text_report))

    print("\n".join(text_report))
    print(f"\n[INFO] Text report saved to {report_path}")

    # Label distribution plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df["final_decision"])
    plt.title("Label Distribution")
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "label_distribution.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"[INFO] Label distribution plot saved to {fig_path}")


if __name__ == "__main__":
    main()