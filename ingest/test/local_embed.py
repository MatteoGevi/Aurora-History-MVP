# ingest/embed.py
import os, json, argparse, hashlib
import requests

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. "https://<resource>.openai.azure.com"
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "text-embedding-3-small")

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def embed_azure(texts, deployment=DEPLOYMENT, endpoint=AZURE_ENDPOINT):
    headers = {
        "api-key": AZURE_OPENAI_KEY,
        "Content-Type": "application/json"
    }
    out = []
    B = 64
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        r = requests.post(
            f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version=2023-05-15",
            headers=headers,
            json={"input": batch}
        )
        r.raise_for_status()
        data = r.json()
        out.extend([d["embedding"] for d in data["data"]])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/out/chunks.paragraphs.jsonl")
    ap.add_argument("--out", default="data/out/chunks.embedded.jsonl")
    args = ap.parse_args()

    rows = list(read_jsonl(args.inp))
    texts = [r.get("content","") for r in rows]

    vecs = embed_azure(texts)

    out_rows = []
    for i, (r, v) in enumerate(zip(rows, vecs)):
        r["id"] = r.get("id") or f"chunk-{i}"
        r["content_hash"] = r.get("content_hash") or sha1(r.get("content",""))
        r["embedding"] = v
        r["embedding_model"] = DEPLOYMENT
        r["embedding_dim"] = len(v)
        out_rows.append(r)

    write_jsonl(args.out, out_rows)
    print(f"Wrote {len(out_rows)} embeddings -> {args.out}")

if __name__ == "__main__":
    main()