import os
import json
from typing import Dict, List

import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def extract_context(example: Dict, include_labels: bool = True, include_meshes: bool = True) -> str:
    """
    PubMedQA 'context' field is a dict with key 'contexts', "labels" and "meshes".
    """
    ctx = example.get("context", {})
    contexts = ctx.get("contexts", []) or []
    labels = ctx.get("labels", []) or []
    meshes = ctx.get("meshes", []) or []
    paras = []
    for i, sent in enumerate(contexts):
        if not isinstance(sent, str):
            continue
        sent = sent.strip()
        if not sent:
            continue

        if include_labels and i < len(labels) and isinstance(labels[i], str):
            label = labels[i].strip()
            if label:
                paras.append(f"{label}: {sent}")
            else:
                paras.append(sent)
        else:
            paras.append(sent)

    text = " ".join(paras)
    
    if include_meshes and meshes:
        mesh_terms = [m.strip() for m in meshes if isinstance(m, str) and m.strip()]
        if mesh_terms:
            mesh_str = "; ".join(mesh_terms)
            text = text + " MeSH terms: " + mesh_str + "."

    return text

class PubMedQARetriever:
    """
    Retriever for PubMedQA vectorstore.

    build_vectorstore.py wrote:
      - index.faiss
      - meta.jsonl (with keys: id, pubid, final_decision, question, is_chunk)
      - stats.json
    """

    def __init__(self, index_dir: str, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.index_dir = index_dir
        self.model_name = model_name

        # 1) load FAISS index + metadata
        self.index = self._load_index()
        self.id_to_meta = self._load_meta()

        # 2) load embedding model for queries
        self.embed_model = SentenceTransformer(self.model_name)

        # 3) pubid -> context (load from pqa_artificial)
        self.pubid_to_example = self._build_pubid_example_map()

    # ---------- private helpers ----------

    def _load_index(self):
        index_path = os.path.join(self.index_dir, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"index.faiss not found in {self.index_dir}")
        print(f"[Retriever] Loading FAISS index from: {index_path}")
        return faiss.read_index(index_path)

    def _load_meta(self) -> Dict[int, dict]:
        meta_path = os.path.join(self.index_dir, "meta.jsonl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.jsonl not found in {self.index_dir}")
        print(f"[Retriever] Loading metadata from: {meta_path}")

        id_to_meta: Dict[int, dict] = {}
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                idx = int(row["id"])
                id_to_meta[idx] = row
        return id_to_meta

    def _build_pubid_example_map(self) -> Dict[str, dict]:
        """
        Build mapping: pubid -> full abstract text.
        """
        print("[Retriever] Loading PubMedQA pqa_artificial for context mapping...")
        ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")

        mapping: Dict[str, dict] = {}
        for ex in ds:
            pubid = ex["pubid"]
            mapping[pubid] = ex

        print(f"[Retriever] Built pubid_to_context map, size = {len(mapping)}")
        return mapping

    def _embed_query(self, question: str):
        q_emb = self.embed_model.encode(
            [f"query: {question}"], 
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        return q_emb

    # ---------- public API ----------
    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        include_labels: bool = True,
        include_meshes: bool = True
    ) -> List[str]:
        """
        Return a list of top-k context texts (abstracts) for the given question.
        """
        q_emb = self._embed_query(question)
        scores, ids = self.index.search(q_emb, top_k * 2)
        ids = ids[0]

        contexts: List[str] = []
        seen_pubids = set()

        for score, idx in zip(scores[0], ids):
            if idx == -1:
                continue

            meta = self.id_to_meta.get(int(idx))
            if not meta:
                continue

            pubid = meta["pubid"]
            if pubid in seen_pubids:
                continue
            seen_pubids.add(pubid)
            
            ex = self.pubid_to_example.get(pubid)
            if not ex:
                continue

            ctx = extract_context(
                ex,
                include_labels=include_labels,
                include_meshes=include_meshes,
            ).strip()
            if not ctx:
                continue

            contexts.append(ctx)

            if len(contexts) >= top_k:
                break

        return contexts

    def retrieve_with_metadata(
        self,
        question: str,
        top_k: int = 5,
        include_labels: bool = True,
        include_meshes: bool = True
    ) -> List[dict]:
        # q_emb: shape (1, dim), float32
        q_emb = self._embed_query(question)
        scores, ids = self.index.search(q_emb, top_k * 2) 

        results: List[dict] = []
        seen_pubids = set()

        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue

            meta = self.id_to_meta.get(int(idx))
            if not meta:
                continue

            pubid = meta["pubid"]
            if pubid in seen_pubids:
                continue
            seen_pubids.add(pubid)
            
            ex = self.pubid_to_example.get(pubid)
            if not ex:
                continue

            ctx = extract_context(
                ex,
                include_labels=include_labels,
                include_meshes=include_meshes,
            ).strip()
            if not ctx:
                continue

            results.append({
                "context": ctx,
                "score": float(score), 
                "pubid": pubid,
                "final_decision": meta.get("final_decision"),
                "artificial_question": meta.get("question"),
            })

            if len(results) >= top_k:
                break

        return results

def get_retriever(index_dir: str, model_name: str = "BAAI/bge-large-en-v1.5") -> PubMedQARetriever:
    return PubMedQARetriever(index_dir=index_dir, model_name=model_name)

dir = "/Users/xinruiwu/Desktop/cs6140/self-aware-adaptive-retrieval/data/index_debug_cpu"
model = "sentence-transformers/all-MiniLM-L6-v2"
retriever = PubMedQARetriever(dir, model)
question = "Do large portion sizes increase bite size and eating rate in overweight women?"

print("\n--- with labels + MeSH ---")
ctxs = retriever.retrieve(
    question,
    top_k=3,
    include_labels=True,
    include_meshes=True,
)
for i, c in enumerate(ctxs, 1):
        print(f"\n[{i}]")
        print(c[-400:], "...\n")
        
print("\n--- without labels / without MeSH ---")
ctxs2 = retriever.retrieve(
    question,
    top_k=3,
    include_labels=False,
    include_meshes=False,
)

for i, c in enumerate(ctxs2, 1):
    print(f"\n[{i}]")
    print(c[-400:], "...\n")

print("Done.")