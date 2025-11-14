import os
import sys
import argparse
from typing import Callable, Iterable, List, Tuple

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from datasets import load_dataset
from ratelimit import limits, sleep_and_retry
from openai import AzureOpenAI

def get_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        return default
    return value


def build_llm_method_local() -> Callable[[str], str]:
    def predict(prompt: str) -> str:
        return "maybe"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PubMedQA yes/no/maybe baseline.")
    parser.add_argument("--model", type=str, default="no-rag", choices=["no-rag"], help="Evaluation model (only 'no-rag' supported).")
    parser.add_argument("--llm", type=str, required=True, choices=["local", "api"], help="LLM backend: local heuristic or Azure API.")
    parser.add_argument("--n", type=int, default=50, help="Number of labeled samples to evaluate (default: 50).")
    parser.add_argument("--deployment", type=str, default=get_env("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano"), help="Azure deployment name (default from env or 'gpt-5-nano').")
    args = parser.parse_args()

    # Load datasets
    print("Loading PubMedQA datasets...")
    labeled_dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    artificial_dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    print(f"labeled_dataset rows: {len(labeled_dataset)}")
    print(f"artificial_dataset rows: {len(artificial_dataset)}")

    # Choose backend
    if args.llm == "local":
        method = build_llm_method_local()
    else:
        try:
            client = init_azure_openai_client()
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        method = build_llm_method_api(client, args.deployment)

    # Evaluate
    n = min(args.n, len(labeled_dataset))
    subset = labeled_dataset.select(range(n))
    y_true, y_pred = run_eval(method, subset)

    accuracy = (np.array(y_true) == np.array(y_pred)).mean() if y_true else 0.0
    hallu = hallucination_rate(y_true, y_pred)
    print(f"\nAccuracy on first {n}: {accuracy:.4f}")
    print(f"Hallucination rate on first {n}: {hallu:.4f}")



