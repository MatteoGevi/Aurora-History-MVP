"""
faiss_test.py
-------------
Build a FAISS index from your chunked JSONL and run a query.

Usage:
    python faiss_test.py --build \
        --chunks data/out/chunks.paragraphs.jsonl \
        --index data/out/index.faiss

    python faiss_test.py --query "Why did Aristotle justify slavery?" \
        --index data/out/index.faiss
"""

import argparse, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def build_index(chunks_file, index_file):
    rows = [json.loads(l) for l in open(chunks_file, encoding="utf-8")]
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [r["content"] for r in rows]
    vecs = model.encode(texts, normalize_embeddings=True)

    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine similarity if vectors are normalized
    index.add(np.array(vecs, dtype="float32"))
    faiss.write_index(index, index_file)

    # save metadata
    for i, r in enumerate(rows):
        r["id"] = i
    with open(index_file + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)

    print(f"Indexed {len(rows)} chunks → {index_file}")

def query_index(query, index_file, k=5):
    index = faiss.read_index(index_file)
    meta = json.load(open(index_file + ".meta.json", encoding="utf-8"))

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    qv = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, k)

    print(f"\nTop-{k} results for query: {query}\n")
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        m = meta[idx]
        print(f"{rank}. score={score:.4f} | section={m.get('section_node_id')} | preview={m['content'][:200]}...\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="Build index from chunks file")
    ap.add_argument("--chunks", default="data/out/chunks.paragraphs.jsonl")
    ap.add_argument("--index", default="data/out/index.faiss")
    ap.add_argument("--query", type=str, help="Run a search query")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    if args.build:
        build_index(args.chunks, args.index)
    if args.query:
        query_index(args.query, args.index, args.k)