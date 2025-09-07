# src/retrieve.py
"""
FAISS retriever with ToC-aware filtering and simple dedup.
Usage:
  python retrieve.py --index data/out/index.faiss --meta data/out/index.faiss.meta.json \
    --query "Why did Aristotle justify slavery?" --k 6 --filter-level-path "Chapter 1"
"""
import argparse, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/out/index.faiss")
    ap.add_argument("--meta", default="data/out/index.faiss.meta.json")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--filter-level-path", default=None)
    args = ap.parse_args()

    index = faiss.read_index(args.index)
    meta = json.load(open(args.meta, "r", encoding="utf-8"))

    candidates = list(range(len(meta)))
    if args.filter_level_path:
        prefix = args.filter_level_path
        candidates = [i for i, m in enumerate(meta) if (m.get("level_path") or "").startswith(prefix)] or candidates

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    qv = model.encode([args.query], normalize_embeddings=True).astype("float32")

    pool_k = min(len(candidates), max(args.k*8, args.k))
    D, I = index.search(qv, pool_k*2)
    raw = [(i, d) for i, d in zip(I[0], D[0]) if i in candidates][:pool_k]

    seen, deduped = set(), []
    for i, d in raw:
        lp = (meta[i].get("level_path") or "")
        if lp in seen: continue
        seen.add(lp); deduped.append((i, d))
        if len(deduped) >= args.k*2: break

    top = deduped[:args.k]
    results = [{
        "rank": r+1,
        "score": float(score),
        "id": meta[i]["id"],
        "page": meta[i].get("page"),
        "level_path": meta[i].get("level_path"),
        "section_node_id": meta[i].get("section_node_id"),
        "preview": (meta[i].get("content") or "")[:320]
    } for r, (i, s
