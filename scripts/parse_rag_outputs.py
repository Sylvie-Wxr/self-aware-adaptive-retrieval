import re
import sys
from pathlib import Path
from collections import defaultdict
import csv

def parse_log(path: str):
    path = Path(path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    results = defaultdict(dict)

    current_setting = None
    include_labels = None
    include_meshes = None
    current_topk = None

    setting_re = re.compile(r"Setting:\s*(\S+)")
    include_re = re.compile(r"include_labels=(True|False),\s*include_meshes=(True|False)")
    topk_re = re.compile(r"--- top_k\s*=\s*(\d+)\s*---")
    acc_re = re.compile(r"Accuracy on first \d+:\s*([0-9.]+)")
    hallu_re = re.compile(r"Hallucination rate on first \d+:\s*([0-9.]+)")

    last_acc = None

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Setting: with_labels_mesh
            m = setting_re.search(line)
            if m:
                current_setting = m.group(1)
                include_labels = None
                include_meshes = None
                current_topk = None
                last_acc = None
                continue

            # include_labels=True, include_meshes=True
            m = include_re.search(line)
            if m:
                include_labels = (m.group(1) == "True")
                include_meshes = (m.group(2) == "True")
                continue

            # topk-n
            m = topk_re.search(line)
            if m:
                current_topk = int(m.group(1))
                last_acc = None
                continue

            # accuracy
            m = acc_re.search(line)
            if m:
                last_acc = float(m.group(1))
                continue

            m = hallu_re.search(line)
            if m and last_acc is not None:
                hallu = float(m.group(1))
                key = (current_setting, include_labels, include_meshes)
                if current_topk is None:
                    continue
                results[key][current_topk] = (last_acc, hallu)
                current_topk = None
                last_acc = None

    return results


def pretty_print(results):
    # results: dict[(setting, include_labels, include_meshes)][top_k] = (acc, hallu)

    for (setting, include_labels, include_meshes) in sorted(results.keys()):
        print("=" * 30)
        print(f"Setting: {setting}")
        print(f"include_labels={include_labels}, include_meshes={include_meshes}")
        print("=" * 30)

        topk_map = results[(setting, include_labels, include_meshes)]
        for k in sorted(topk_map.keys()):
            acc, hallu = topk_map[k]
            print(f"top-k = {k}: Accuracy = {acc:.4f}, Hallucination rate = {hallu:.4f}")
        print()

def write_csv(results, out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["setting", "include_labels", "include_meshes", "top_k", "accuracy", "hallucination"])

        for (setting, include_labels, include_meshes), topk_map in results.items():
            for k, (acc, hallu) in sorted(topk_map.items()):
                writer.writerow([
                    setting,
                    include_labels,
                    include_meshes,
                    k,
                    acc,
                    hallu,
                ])

    print(f"[CSV] Results written to: {out_path}")

def main():
    log_path = "/home/wu.xinrui/ondemand/dev/self-aware-adaptive-retrieval/logs/output_experiments_1000.txt"

    results = parse_log(log_path)
    pretty_print(results)

    csv_path = "/home/wu.xinrui/ondemand/dev/self-aware-adaptive-retrieval/logs/eval_results.csv"
    write_csv(results, csv_path)



if __name__ == "__main__":
    main()
