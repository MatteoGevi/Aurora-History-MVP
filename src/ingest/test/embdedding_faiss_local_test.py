# scripts/embed_faiss_local.py
# pip install faiss-cpu sentence-transformers
import json
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

chunks = [json.loads(l) for l in open("data/out/chunks.paragraphs.jsonl","r")]
texts = [c["content"] for c in chunks]
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
X = model.encode(texts, normalize_embeddings=True).astype("float32")

index = faiss.IndexFlatIP(X.shape[1])  # cosine with normalized vectors == dot product
index.add(X)

def search(query, k=8):
    qv = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(qv, k)
    return [(float(scores[0][i]), chunks[int(idxs[0][i])]) for i in range(k)]