import os
import sys
import argparse
from typing import Callable, Iterable, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import logging
import time
from vllm import LLM, SamplingParams

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from datasets import load_dataset
from ratelimit import limits, sleep_and_retry
from openai import AzureOpenAI

from retrieve import PubMedQARetriever, get_retriever

DEFAULT_RESULTS_BASE = "/home/wu.xinrui/ondemand/dev/self-aware-adaptive-retrieval/results"

def get_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        return default
    return value


def build_llm_method_local(
    llm: LLM, 
    sampling_params: SamplingParams,
    retriever: Optional[PubMedQARetriever] = None,
    top_k: int = 3,
    include_labels: bool = True,
    include_meshes: bool = True,
) -> Callable[[str], str]:
    def predict(question: str) -> str:
        context_block = ""
        if retriever is not None:
            contexts = retriever.retrieve(
                question, 
                top_k=top_k, 
                include_labels=include_labels,
                include_meshes=include_meshes)
            if contexts:
                joined = "\n\n---\n\n".join(contexts)
                context_block = (
                    "You are given several PubMed article abstracts.\n"
                    "Answer the question ONLY based on these abstracts.\n"
                    "If the information is insufficient, answer 'maybe'.\n\n"
                    f"Contexts:\n{joined}\n\n"
                )
        prompt = (
            "You must answer strictly with one label: yes, no, or maybe.\n"
            "No explanations. Only output the label.\n\n"
        )
        if context_block:
            prompt += context_block

        prompt += f"Question: {question}\nAnswer:"

        output = llm.generate([prompt], sampling_params)
        return output[0].outputs[0].text.strip().lower()
    return predict

def build_adaptive_method_local(
    llm: LLM,
    sampling_params: SamplingParams,
    retriever: PubMedQARetriever,
    top_k: int = 3,
    include_labels: bool = True,
    include_meshes: bool = True,
    return_meta: bool = False,
) -> Callable[[str], str]:
    """
    Adaptive RAG (local), simple:

    - confidence == "high"
    - confidence != "high"

    return_meta=False: final label
    return_meta=True:  (final, meta)
    """

    def predict(question: str):
        t0 = time.time()

        # ---------- Step 1: self-assessment ----------
        t_self_start = time.time()
        first_answer, confidence = self_assess_local(llm, question)
        t_self_ms = (time.time() - t_self_start) * 1000.0

        used_retrieval = False
        t_retrieve_ms = 0.0
        final = first_answer

        # ---------- Step 2: decision ----------
        if confidence != "high":
            # Confidence not high →  RAG
            used_retrieval = True
            t_ret_start = time.time()
            contexts = retriever.retrieve(
                question,
                top_k=top_k,
                include_labels=include_labels,
                include_meshes=include_meshes,
            )
            t_retrieve_ms = (time.time() - t_ret_start) * 1000.0

            if contexts:
                joined = "\n\n---\n\n".join(contexts)
                context_block = (
                    "You are given several PubMed article abstracts.\n"
                    "Answer ONLY using these abstracts.\n"
                    "If insufficient, answer 'maybe'.\n\n"
                    f"Contexts:\n{joined}\n\n"
                )

                prompt = (
                    "You must answer strictly: yes, no, or maybe.\n"
                    "No explanations.\n\n"
                )
                prompt += context_block
                prompt += f"Question: {question}\nAnswer:"

                out = llm.generate([prompt], sampling_params)
                raw = (out[0].outputs[0].text or "").strip().lower()

                if raw not in {"yes", "no", "maybe"}:
                    low = raw.lower()
                    if "yes" in low:
                        final = "yes"
                    elif "no" in low:
                        final = "no"
                    elif "maybe" in low:
                        final = "maybe"
                    else:
                        final = "maybe"
                else:
                    final = raw
            else:
                # If cannot retrieve documents → return self-answer
                final = first_answer
        else:
            # confidence == "high" → self-answer
            final = first_answer

        t_total_ms = (time.time() - t0) * 1000.0

        if not return_meta:
            return final

        meta = {
            "first_answer": first_answer,
            "confidence": confidence,
            "used_retrieval": used_retrieval,
            "t_self_ms": t_self_ms,
            "t_retrieve_ms": t_retrieve_ms,
            "t_total_ms": t_total_ms,
        }
        return final, meta

    return predict



def init_azure_openai_client() -> AzureOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not azure_endpoint:
        raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT environment variable.")
    api_version = get_env("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )


def build_llm_method_api(client: AzureOpenAI, deployment: str) -> Callable[[str], str]:
    """
    Azure OpenAI backend with simple rate limit.
    """
    @sleep_and_retry
    @limits(calls=100, period=60)
    def get_llm_response(prompt: str) -> str:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Answer ONLY one of: yes, no, maybe. No explanations. You are a medical professional."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=16384,
        )
        msg = response.choices[0].message
        content = (msg.content or "").strip()

        if not content:
            print("DEBUG raw message:", msg)

        return content
    return get_llm_response


def run_eval(method: Callable[[str], str], data: Iterable[dict]) -> Tuple[List[str], List[str]]:
    y_true, y_pred = [], []
    for i, ex in enumerate(data, 1):
        gold = (ex["final_decision"] or "").strip().lower()
        pred = (method(ex["question"]) or "").strip().lower()
        y_true.append(gold)
        y_pred.append(pred)
        print(f"Q{i}: predict={pred} | gold={gold}")
    return y_true, y_pred

def hallucination_rate(y_true: List[str], y_pred: List[str]) -> float:
    """
    Overclaim: truth==maybe AND pred in {yes,no}
    Contradiction: truth in {yes,no} AND pred in {yes,no} and pred != truth
    Safe miss: truth in {yes,no} AND pred==maybe (not hallucination)
    """
    total = len(y_true)
    hallucinations = 0
    for truth, pred in zip(y_true, y_pred):
        t = (truth or "").strip().lower()
        p = (pred or "").strip().lower()
        if t == "maybe" and p in {"yes", "no"}:
            hallucinations += 1
        elif t in {"yes", "no"} and p in {"yes", "no"} and p != t:
            hallucinations += 1
    return hallucinations / total if total > 0 else 0.0


def run_rag_eval(
    llm: LLM,
    sampling_params: SamplingParams,
    retriever,
    subset,
    n: int,
    results_dir: str
) -> None:
    """local + rag-always: sweep top_k / labels+MeSH, 
    Write to CSV + Curve + best confusion matrix."""
    topk_values = [1, 2, 3, 4, 5]
    settings = [
        ("with_labels_mesh", True, True),
        ("no_labels_mesh", False, False),
    ]

    sweep_path = os.path.join(results_dir, "rag_sweep_metrics.csv")
    conf_path = os.path.join(results_dir, "rag_confusions.csv")
    
    with open(sweep_path, "w") as f_metrics, open(conf_path, "w") as f_conf:
        # CSV header: metrics
        f_metrics.write(
            "setting_name,include_labels,include_meshes,top_k,accuracy,hallucination\n"
        )
        
        # CSV header: confusion matrix (flattened 3x3)
        f_conf.write(
            "setting_name,include_labels,include_meshes,top_k,"
            "cm_yy,cm_yn,cm_ym,cm_ny,cm_nn,cm_nm,cm_my,cm_mn,cm_mm\n"
        )

        labels = ["yes", "no", "maybe"]

        for setting_name, include_labels, include_meshes in settings:
            print("\n==============================")
            print(f"Setting: {setting_name}")
            print(f"include_labels={include_labels}, include_meshes={include_meshes}")
            print("==============================")

            for k in topk_values:
                print(f"\n--- top_k = {k} ---")

                method_for_config = build_llm_method_local(
                    llm,
                    sampling_params,
                    retriever=retriever,
                    top_k=k,
                    include_labels=include_labels,
                    include_meshes=include_meshes,
                )

                y_true, y_pred = run_eval(method_for_config, subset)
                accuracy = (np.array(y_true) == np.array(y_pred)).mean() if y_true else 0.0
                hallu = hallucination_rate(y_true, y_pred)

                # Write to CSV
                f_metrics.write(
                    f"{setting_name},{int(include_labels)},{int(include_meshes)},"
                    f"{k},{accuracy:.6f},{hallu:.6f}\n"
                )

                print(f"Accuracy on first {n}: {accuracy:.4f}")
                print(f"Hallucination rate on first {n}: {hallu:.4f}")

                # Caluculate and write confusion matrix（3x3）
                cm = confusion_matrix(y_true, y_pred, labels=labels)

                # Order: [yy, yn, ym, ny, nn, nm, my, mn, mm]
                cm_flat = cm.flatten()  # row-major,  yes row first，then no row，next maybe row
                f_conf.write(
                    f"{setting_name},{int(include_labels)},{int(include_meshes)},{k},"
                    + ",".join(str(int(x)) for x in cm_flat)
                    + "\n"
                )

    print(f"[INFO] RAG sweep metrics saved to {sweep_path}")

def run_baseline_eval(
    method: Callable[[str], str],
    subset,
    n: int,
    args: argparse.Namespace,
    results_dir: str,  
) -> None:
    """no-rag local or API baseline: Run once + confusion matrix + metrics file。"""
    y_true, y_pred = run_eval(method, subset)
    accuracy = (np.array(y_true) == np.array(y_pred)).mean() if y_true else 0.0
    hallu = hallucination_rate(y_true, y_pred)

    print(f"\nAccuracy on first {n}: {accuracy:.4f}")
    print(f"Hallucination rate on first {n}: {hallu:.4f}")

    # Confusion matrix
    labels = ["yes", "no", "maybe"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    plt.figure()
    disp.plot(values_format="d", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({args.model}, {args.llm})")
    plt.tight_layout()
    cm_path = os.path.join(results_dir, f"confusion_matrix_{args.model}_{args.llm}.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"[INFO] Confusion matrix saved to {cm_path}")

    # Save baseline metrics
    baseline_path = os.path.join(results_dir, f"baseline_metrics_{args.model}_{args.llm}.txt")
    with open(baseline_path, "w") as f:
        f.write(f"model={args.model}\n")
        f.write(f"llm={args.llm}\n")
        f.write(f"n={n}\n")
        f.write(f"accuracy={accuracy:.6f}\n")
        f.write(f"hallucination={hallu:.6f}\n")

    print(f"[INFO] Baseline metrics saved to {baseline_path}")
    
def run_adaptive_rag_eval(
    llm: LLM,
    sampling_params: SamplingParams,
    retriever: PubMedQARetriever,
    subset,
    n: int,
    results_dir: str,
) -> None:
    """
    local + adaptive-rag: sweep top_k / labels+MeSH
    record accuracy / hallucination / decision and latency to csv
    """
    topk_values = [1, 2, 3, 4, 5]
    settings = [
        ("with_labels_mesh", True, True),
        ("no_labels_mesh", False, False),
    ]

    metrics_path = os.path.join(results_dir, "adaptive_sweep_metrics.csv")
    decisions_path = os.path.join(results_dir, "adaptive_decisions.csv")

    with open(metrics_path, "w") as f_metrics, open(decisions_path, "w") as f_dec:
        # setting header
        f_metrics.write(
            "setting_name,include_labels,include_meshes,top_k,"
            "accuracy,hallucination,used_retrieval_rate,"
            "avg_t_self_ms,avg_t_retrieve_ms,avg_t_total_ms\n"
        )
        # decision and latency
        f_dec.write(
            "setting_name,include_labels,include_meshes,top_k,idx,"
            "gold,pred,first_answer,confidence,used_retrieval,"
            "t_self_ms,t_retrieve_ms,t_total_ms\n"
        )

        labels = ["yes", "no", "maybe"]

        for setting_name, include_labels, include_meshes in settings:
            print("\n==============================")
            print(f"[Adaptive] Setting: {setting_name}")
            print(f"include_labels={include_labels}, include_meshes={include_meshes}")
            print("==============================")

            for k in topk_values:
                print(f"\n--- [Adaptive] top_k = {k} ---")

                # adaptive method
                method = build_adaptive_method_local(
                    llm,
                    sampling_params,
                    retriever=retriever,
                    top_k=k,
                    include_labels=include_labels,
                    include_meshes=include_meshes,
                    return_meta=True,
                )

                y_true, y_pred = [], []
                hallu_flags = []
                used_retrieval_flags = []
                t_self_list, t_ret_list, t_total_list = [], [], []

                for i, ex in enumerate(subset, 1):
                    gold = (ex["final_decision"] or "").strip().lower()
                    pred, meta = method(ex["question"])

                    y_true.append(gold)
                    y_pred.append(pred)

                    # hallucination
                    is_hallu = 0
                    g, p = gold, pred
                    if g == "maybe" and p in {"yes", "no"}:
                        is_hallu = 1
                    elif g in {"yes", "no"} and p in {"yes", "no"} and g != p:
                        is_hallu = 1
                    hallu_flags.append(is_hallu)

                    used_retrieval_flags.append(int(meta["used_retrieval"]))
                    t_self_list.append(meta["t_self_ms"])
                    t_ret_list.append(meta["t_retrieve_ms"])
                    t_total_list.append(meta["t_total_ms"])

                    f_dec.write(
                        f"{setting_name},{int(include_labels)},{int(include_meshes)},"
                        f"{k},{i},{gold},{pred},{meta['first_answer']},{meta['confidence']},"
                        f"{int(meta['used_retrieval'])},"
                        f"{meta['t_self_ms']:.3f},{meta['t_retrieve_ms']:.3f},{meta['t_total_ms']:.3f}\n"
                    )

                    print(
                        f"Q{i}: pred={pred} | gold={gold} | "
                        f"first={meta['first_answer']} | conf={meta['confidence']} | "
                        f"used_retr={meta['used_retrieval']}"
                    )

                y_true_arr = np.array(y_true)
                y_pred_arr = np.array(y_pred)
                accuracy = (y_true_arr == y_pred_arr).mean() if y_true else 0.0
                hallu_rate = np.mean(hallu_flags) if hallu_flags else 0.0
                used_retrieval_rate = np.mean(used_retrieval_flags) if used_retrieval_flags else 0.0

                avg_t_self = float(np.mean(t_self_list)) if t_self_list else 0.0
                avg_t_ret = float(np.mean(t_ret_list)) if t_ret_list else 0.0
                avg_t_total = float(np.mean(t_total_list)) if t_total_list else 0.0

                print(f"[Adaptive] Accuracy on first {n}: {accuracy:.4f}")
                print(f"[Adaptive] Hallucination rate: {hallu_rate:.4f}")
                print(f"[Adaptive] Used retrieval on {used_retrieval_rate*100:.1f}% questions")
                print(f"[Adaptive] avg_t_self={avg_t_self:.1f}ms, "
                      f"avg_t_retrieve={avg_t_ret:.1f}ms, avg_t_total={avg_t_total:.1f}ms")

                f_metrics.write(
                    f"{setting_name},{int(include_labels)},{int(include_meshes)},"
                    f"{k},{accuracy:.6f},{hallu_rate:.6f},{used_retrieval_rate:.6f},"
                    f"{avg_t_self:.3f},{avg_t_ret:.3f},{avg_t_total:.3f}\n"
                )

    print(f"[INFO] Adaptive sweep metrics saved to {metrics_path}")
    print(f"[INFO] Adaptive decision logs saved to {decisions_path}")
    
def self_assess_local(llm: LLM, question: str) -> tuple[str, str]:
    """
    return: (answer, confidence)
    answer ∈ {"yes", "no", "maybe"}
    confidence ∈ {"high", "medium", "low"}
    """
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
    )
    prompt = f"""
        You are a medical QA assistant answering PubMed-style clinical questions.

        1. First, answer the question with exactly one of: "yes", "no", or "maybe".
        2. Then, rate your confidence in that answer as one of: "high", "medium", "low".

        Return your result strictly as a JSON object, for example:
        {{"answer": "yes", "confidence": "medium"}}

        Question: {question}
    """
    out = llm.generate([prompt], sampling_params)
    text = out[0].outputs[0].text.strip()

    # find JSON
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]
        obj = json.loads(json_str)
        ans = (obj.get("answer") or "").strip().lower()
        conf = (obj.get("confidence") or "").strip().lower()
    except Exception:
        ans = "maybe"
        for cand in ("yes", "no", "maybe"):
            if cand in text.lower():
                ans = cand
                break
        conf = "medium"
        for cand in ("high", "medium", "low"):
            if cand in text.lower():
                conf = cand
                break

    if ans not in {"yes", "no", "maybe"}:
        ans = "maybe"
    if conf not in {"high", "medium", "low"}:
        conf = "medium"

    return ans, conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PubMedQA yes/no/maybe baseline.")
    parser.add_argument("--model", type=str, default="no-rag", choices=["no-rag", "rag-always", "adaptive-rag"], help="Evaluation model: no-rag, rag-always or adaptive-rag.")
    parser.add_argument("--llm", type=str, required=True, choices=["local", "api"], help="LLM backend: local heuristic or Azure API.")
    parser.add_argument("--n", type=int, default=50, help="Number of labeled samples to evaluate (default: 50).")
    parser.add_argument("--deployment", type=str, default=get_env("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano"), help="Azure deployment name (default from env or 'gpt-5-nano').")
    parser.add_argument(
    "--index_dir",
    type=str,
    default="/projects/insightx-lab/xinruiwu/pubmedqa_index_full_bge",
    help="Directory of FAISS index (index.faiss + meta.jsonl) for RAG.",
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Embedding model name used when building the index.",
    )
    parser.add_argument(
        "--rag_top_k",
        type=int,
        default=3,
        help="Top-k contexts to retrieve when using RAG.",
    )
    parser.add_argument(
    "--results-dir",
    type=str,
    default=DEFAULT_RESULTS_BASE,
    help="Base directory to store all evaluation outputs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for this run; if not set, a timestamp-based name will be used.",
    )
    args = parser.parse_args()
    
    # Generate run_time
    if args.run_name is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{ts}_{args.llm}_{args.model}_n{args.n}"

    RESULTS_DIR = os.path.join(args.results_dir, args.run_name)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save run config to json
    config = vars(args).copy()

    # If using local LLM, deployment is irrelevant → remove it for cleaner logs
    if args.llm == "local":
        config.pop("deployment", None)

    config_path = os.path.join(RESULTS_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[INFO] Saved run config to {config_path}")

    
    # Init logger
    log_path = os.path.join(RESULTS_DIR, "stdout.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),  # print to terminal
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting evaluation with args: {args}")

    # Load datasets
    print("Loading PubMedQA datasets...")
    labeled_dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    print(f"labeled_dataset rows: {len(labeled_dataset)}")
    
    
    # Decide whether to use retriever (local + rag-always / adaptive)
    need_retriever = (args.llm == "local") and (args.model in {"rag-always", "adaptive-rag"})
    retriever: Optional[PubMedQARetriever] = None
    if need_retriever:
        print(f"[Eval] Using RAG with index_dir={args.index_dir}, retriever_model={args.retriever_model}")
        retriever = get_retriever(index_dir=args.index_dir, model_name=args.retriever_model)
    else:
        print("[Eval] Running no-rag baseline (question only).")
        
    # Evaluate
    n = min(args.n, len(labeled_dataset))
    subset = labeled_dataset.select(range(n))

    # Choose backend
    if args.llm == "local":
        try:
            llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.3")
            sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            stop=["\n", ".", ","],
        )

        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
            
        if args.model == "rag-always":
            if retriever is None:
                raise RuntimeError("rag-always requires a retriever.")
            # local + rag-always → RAG sweep
            run_rag_eval(llm, sampling_params, retriever, subset, n, RESULTS_DIR)
        
        elif args.model == "adaptive-rag":
        # adaptive rag
            if retriever is None:
                raise RuntimeError("adaptive-rag model requires a retriever.")
            run_adaptive_rag_eval(llm, sampling_params, retriever, subset, n, RESULTS_DIR)
            
        else:
            # local + no-rag baseline
            method = build_llm_method_local(
                llm, 
                sampling_params,
                retriever=None,
                top_k=args.rag_top_k,
                include_labels=True,
                include_meshes=True)
            run_baseline_eval(method, subset, n, args, RESULTS_DIR)
    
    # API base line, no rag
    else:
        try:
            client = init_azure_openai_client()
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        method = build_llm_method_api(client, args.deployment)
        run_baseline_eval(method, subset, n, args)




