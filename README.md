# Self-Aware Adaptive Retrieval

This repository implements **No-RAG**, **RAG-Always**, and (coming soon) **Adaptive RAG** strategies for PubMedQA using a local LLM and FAISS-based retrieval.

---

### 1. Repository Structure

```
self-aware-adaptive-retrieval/
│
├── build_vectorstore.py          # build embeddings + FAISS index
├── retrieve.py                   # retrieval utils
├── evaluation.py                 # No-RAG & RAG-Always eval
├── adaptive_eval.py              # Adaptive-RAG eval
│
├── scripts/
│   ├── data_check.py             # dataset stats
│   ├── context_length_stats.py   # question length stats
│   ├── plot_rag_always.py        # RAG-Always plots
│   ├── plot_adaptive.py          # Adaptive-RAG plots
│   └── plot_three_methods.py     # compare 3 methods
│
└── results/
    ├── data_eda/                 # dataset analysis
    │   ├── data_overview.txt
    │   └── label_distribution.png
    │
    ├── no_rag_local_xxx/         # No-RAG outputs
    │
    ├── YYYYMMDD_rag-always_xxx/  # RAG-Always outputs
    │
    ├── YYYYMMDD_adaptive_xxx/    # Adaptive-RAG outputs
    │
    └── three_methods_plots/    # 3-method comparison


```

---

### 2. Environment Setup (UV + GPU Cluster)

**2.1 Create environment**

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

**2.2 Install dependencies**

```bash
uv pip install uv
uv pip install vllm --torch-backend=auto
uv pip install sklearn matplotlib numpy datasets ratelimit openai sentence_transformers
uv pip install faiss-cpu faiss-gpu-cu12
```

---

### 3. Dataset Checking

**3.1 Run data check**

```bash
cd scripts
uv run python data_check.py
```

Outputs saved to:

```
results/data_eda/
```

---

### 4. Building the Vector Store (Embeddings + FAISS)

**4.1 Run vector store builder**

```bash
uv run python build_vectorstore.py
```

> This step may take a long time in interactive cluster mode.

Prebuilt index is available under:

```
/projects/insightx-lab/xinruiwu
```

---

### 5. Running Evaluations

---

### 5.1 Run No-RAG Baseline

```bash
uv run python evaluation.py --llm local --model no-rag --n 1000   --run-name no_rag_local_1000_01 2>&1 | tee logs/no_rag_local_1000_01.log
```

### Output files include:

- baseline_metrics_no-rag.txt
- confusion_matrix_no-rag_local.png
- config.json
- stdout.log

---

### 5.2 Run RAG-Always

```bash
uv run python evaluation.py --llm local --model rag-always --n 1000   2>&1 | tee logs/rag_always_1000_04.log
```

### Output files include:

- rag_sweep_metrics.csv
- rag_confusions.csv
- config.json
- stdout.log

---

### 5.3 Run Adaptive-RAG

```bash
uv run python evaluation.py --llm local --model rag-adaptive --n 1000   2>&1 | tee logs/rag_adaptive_1000_with_latency.log
```

### Output files include:

- rag_sweep_metrics.csv
- adaptive_decisions.csv
- config.json
- stdout.log

---

### Optinally: 5.4 Test with api endpoint (kept for local test)

```bash
uv run python evaluation.py --llm api --model no-rag --n 1000   2>&1 | tee logs/no_rag_api_1000_04.log
```

API will be quite slow, and not much use now. It may be removed in the final update.

---

### 6. Plotting RAG-Always Results

```bash
cd scripts
uv run python plot_rag_always.py --csv /path/to/rag_sweep_metrics.csv --conf-csv /path/to/rag_confusions.csv
```

Generates:

- Accuracy curve
- Hallucination curve
- Best-config confusion matrix

---

### 7. Plotting RAG-Adaptive Results

```bash
cd scripts
uv run python plot_adaptive.py \
  --csv /path/to/adaptive_metrics.csv \
  --decisions-csv /path/to/adaptive_decisions.csv
```

Generates:

- Accuracy vs top_k (for with_labels_mesh vs no_labels_mesh)
- Hallucination vs top_k
- Used-retrieval rate vs top_k
- Latency vs top_k (self-assessment, retrieval, end-to-end)
- Confusion matrices for:
    - best hallucination config
    - best accuracy config

---

### 8. Plotting three method comparisons

```bash
cd scripts
uv run python plot_three_methods.py \
  --baseline /path/to/baseline_metrics.txt \
  --rag /path/to/rag_sweep_metrics.csv \
  --adaptive /path/to/adaptive_metrics.csv \
  --out-dir /path/to/output_dir

```

Generates:

- accuracy_vs_latency on three methods
- hallucination_vs_latency on three methods
- metrics_comparison_bar on three methods
- acc_vs_hallu_scatter on three methods

---

### 9. Script Descriptions

**scripts/data_check.py**
- Runs missing-value detection  
- Outputs label distribution  
- Saves plots and text summary  

**scripts/context_length_stats.py**
- Computes context word-length  
- Stats (median ≈ 200, p90 ≈ 258) show chunking is unnecessary  

**scripts/plot_rag_always.py**
- Reads CSV from RAG-Always  
- Plots accuracy/hallucination curves  
- Generates confusion matrix for best configuration  

**scripts/plot_adaptive.py**
- Reads adaptive_metrics.csv for Adaptive-RAG sweeps
- Plots accuracy & hallucination vs top_k for with_labels_mesh and no_labels_mesh
- Plots used-retrieval rate vs top_k
- Plots latency breakdown vs top_k (self, retrieve, total)
- Uses adaptive_decisions.csv to build confusion matrices for:
    - lowest-hallucination config
    - highest-accuracy config

**scripts/plot_three_methods.py**
- Reads baseline, RAG-Always, and Adaptive-RAG metrics
- Selects the best RAG-Always / Adaptive configurations (lowest hallucination, then highest accuracy)
- Plots latency–accuracy, latency–hallucination, bar chart, and accuracy–hallucination scatter for the three methods

---

### 8. Output Folder Layout

```
results/
│
├── data_eda/
│   ├── data_overview.txt
│   └── label_distribution.png
│
├── no_rag_local_1000_01/
│   ├── baseline_metrics_no-rag.txt
│   ├── confusion_matrix_no-rag_local.png
│   ├── config.json
│   └── stdout.log
│
├── YYYYMMDD_local_rag-always_1000/
│   ├── rag_sweep_metrics.csv
│   ├── rag_confusions.csv
│   │
│   ├── rag_accuracy_curve.png
│   ├── rag_hallucination_curve.png
│   ├── rag_confusion_best.png
│   │
│   ├── config.json
│   └── stdout.log
│
├── YYYYMMDD_local_adaptive_1000/
│   ├── adaptive_metrics.csv
│   ├── adaptive_decisions.csv
│   │
│   ├── adaptive_acc_hallu_vs_topk.png
│   ├── adaptive_used_retrieval_vs_topk.png
│   ├── adaptive_latency_vs_topk.png
│   ├── adaptive_confusion_best_hallucination.png
│   ├── adaptive_confusion_best_accuracy.png
│   │
│   ├── config.json
│   └── stdout.log
│
└── three_method_plots/
    ├── accuracy_vs_latency.png
    ├── hallucination_vs_latency.png
    ├── metrics_comparison_bar.png
    ├── acc_vs_hallu_scatter.png

```

---

### 9. Notes

- All evaluations support arbitrary --n  
- Vector store needs to be built only once  
- Logging via `tee` is optional but recommended  


