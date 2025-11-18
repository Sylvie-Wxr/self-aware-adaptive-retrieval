import os
import sys
import json
import argparse
from typing import List, Dict, Tuple

import numpy as np
from datasets import load_dataset

# --- FAISS ---
try:
    import faiss
except ImportError:
    print("ERROR: faiss is required. Install via: pip install faiss-cpu", file=sys.stderr)
    raise

# --- Embedding model ---
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers required. Install via: pip install sentence-transformers", file=sys.stderr)
    raise



#  Safe batching
def batched(iterable, batch_size):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# --------------------------------------------------------------
#  Step 1: Load PubMedQA + extract context text
# --------------------------------------------------------------
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



# --------------------------------------------------------------
#  Text chunking (default disabled)
# --------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 256, overlap: int = 32) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# --------------------------------------------------------------
#  Main build function
# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build FAISS vectorstore for PubMedQA pqa_artificial using BGE.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output folder to save index.faiss + meta.jsonl")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Embedding batch size (A100: 256, V100: 128)")
    parser.add_argument("--model", type=str, default="BAAI/bge-large-en-v1.5",
                        help="Embedding model name")
    parser.add_argument("--chunk", action="store_true",
                        help="Enable chunking of long abstracts")
    parser.add_argument("--chunk_size", type=int, default=256,
                        help="Chunk word count (only if --chunk enabled)")
    parser.add_argument("--overlap", type=int, default=32,
                        help="Word overlap between chunks")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Optional limit for debugging (e.g. 5000). -1 for full dataset")
    parser.add_argument("--test_only", action="store_true",
                        help="Only test model & FAISS pipeline on 100 samples")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    index_path = os.path.join(args.out_dir, "index.faiss")
    meta_path = os.path.join(args.out_dir, "meta.jsonl")
    stats_path = os.path.join(args.out_dir, "stats.json")

    # -------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------
    print("Loading PubMedQA pqa_artificial/train ...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")

    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Total rows to process: {len(ds)}")

    # -------------------------------------------------------------------
    # Load embedding model
    # -------------------------------------------------------------------
    print(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    # For BGE-large-en-v1.5 dimension = 1024
    sample_emb = model.encode(["test"], normalize_embeddings=True)
    embed_dim = sample_emb.shape[1]
    print(f"Embedding dimension: {embed_dim}")

    # Prepare FAISS index
    index = faiss.IndexFlatIP(embed_dim)

    # -------------------------------------------------------------------
    # If test only: do a quick 100-sample test, local dry-run
    # -------------------------------------------------------------------
    if args.test_only:
        print("Running quick test on 100 samples...")
        small_ds = ds.select(range(100))
        texts = []
        metas = []

        for ex in small_ds:
            txt = extract_context(ex, include_labels=True, include_meshes=True)
            if not txt:
                continue
            texts.append("passage: " + txt)
            metas.append({
                "pubid": ex["pubid"],
                "final_decision": ex["final_decision"],
                "question": ex["question"]
            })

        test_emb = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        print("Test embedding shape:", test_emb.shape)

        index.add(test_emb)
        test_question = metas[0]["question"]
        test_q_emb = model.encode(
            [f"query: {test_question}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        print("\nFAISS test search with query embedding:")
        scores, ids = index.search(test_q_emb, 3)
        print("Test question:", test_question[:100], "\n")
        print("scores:", scores)
        print("ids:", ids)

        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
            if idx == -1:
                continue
            meta = metas[idx]
            print(f"\n{rank}. Score: {score:.4f}")
            print("   PubID:", meta["pubid"])
            print("   Question snippet:", meta["question"][:100], "...")

        print("\nTest complete. No files were saved because --test_only was used.")
        return

    # -------------------------------------------------------------------
    # Full pass: embedding & indexing
    # -------------------------------------------------------------------
    print("Building vectorstore...")
    # Open metadata file
    meta_file = open(meta_path, "w", encoding="utf-8")

    # Counters
    total_vectors = 0
    total_passages = 0


    for batch in batched(ds, args.batch_size):
        texts = []
        metas = []

        for ex in batch:
            txt = extract_context(ex, include_labels=True, include_meshes=True)
            if not txt:
                continue

            # Optional chunking
            if args.chunk:
                chunks = chunk_text(txt, args.chunk_size, args.overlap)
                for c in chunks:
                    texts.append("passage: " + c)
                    metas.append({
                        "pubid": ex["pubid"],
                        "final_decision": ex["final_decision"],
                        "question": ex["question"],
                        "is_chunk": True
                    })
                    total_passages += 1
            else:
                texts.append("passage: " + txt)
                metas.append({
                    "pubid": ex["pubid"],
                    "final_decision": ex["final_decision"],
                    "question": ex["question"],
                    "is_chunk": False
                })
                total_passages += 1

        if not texts:
            continue

        # Encode
        emb = model.encode(
            texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype("float32")

        # Add to FAISS
        index.add(emb)

        # Save metadata with aligned ids
        for i, meta in enumerate(metas):
            meta_row = {"id": total_vectors + i, **meta}
            meta_file.write(json.dumps(meta_row) + "\n")

        total_vectors += len(metas)

    # -------------------------------------------------------------------
    # Save index + stats
    # -------------------------------------------------------------------
    print(f"\nTotal vectors added: {total_vectors}")
    print(f"Saving FAISS index to: {index_path}")
    faiss.write_index(index, index_path)

    meta_file.close()

    stats = {
        "total_vectors": total_vectors,
        "embedding_dim": embed_dim,
        "chunk_enabled": args.chunk,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "dataset_rows": len(ds)
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Metadata saved to: {meta_path}")
    print(f"Stats saved to: {stats_path}")
    print("DONE.")


if __name__ == "__main__":
    main()
