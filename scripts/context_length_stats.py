from datasets import load_dataset
import numpy as np

def count_words(text):
    return len(text.split())

def main():
    print("Loading pqa_artificial...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")

    lengths = []
    for ex in ds:
        ctx_list = ex["context"]["contexts"] or []
        text = " ".join(ctx_list)
        lengths.append(count_words(text))

    print("Total samples:", len(lengths))
    print("Avg:", np.mean(lengths))
    print("Median:", np.median(lengths))
    print("p90:", np.percentile(lengths, 90))
    print("p95:", np.percentile(lengths, 95))
    print("Max:", np.max(lengths))
    print("Min:", np.min(lengths))

if __name__ == "__main__":
    main()
