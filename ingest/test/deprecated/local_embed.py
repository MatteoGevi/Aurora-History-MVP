from __future__ import annotations
import argparse, json, re, statistics, random, hashlib
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------------------------- CLI ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Embedding sanity check for chunked textbook.")
    ap.add_argument("--chunks", type=Path, required=True,
                    help="Path to chunks.paragraphs.jsonl")
    ap.add_argument("--sample", type=int, default=500,
                    help="Sample size for embedding checks (default 500)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    return ap.parse_args()


# ---------------------------- Utils ----------------------------
def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def stats_print(df: pd.DataFrame) -> None:
    lengths = df["text"].str.len()
    print(
        f"n: {len(df)} | "
        f"avg chars: {lengths.mean():.1f} | "
        f"median: {int(lengths.median())} | "
        f">2000 chars: {(lengths>2000).sum()} | "
        f"empty: {(lengths==0).sum()}"
    )

def duplicate_report(df: pd.DataFrame) -> None:
    # Exact normalized duplicate count (full text)
    hashes = df["text"].map(lambda t: hashlib.sha1(normalize_text(t).encode()).hexdigest())
    dup_count = hashes.duplicated().sum()
    print(f"exact duplicates (normalized full text): {dup_count}")

    # Show a few repeated snippets, if any
    vc = df["text"].value_counts()
    reps = vc[vc > 1]
    if not reps.empty:
        print("Top repeated snippets:")
        for txt, cnt in reps.head(3).items():
            prev = txt.replace("\n"," ")[:160]
            print(f"  count={cnt} | “{prev}…”")


def embed_sanity(df: pd.DataFrame, sample_n: int, seed: int = 42) -> None:
    if len(df) == 0:
        print("No rows to embed.")
        return

    random.seed(seed)
    idx = list(range(len(df)))
    random.shuffle(idx)
    idx = idx[: min(sample_n, len(df))]

    sub = df.iloc[idx].copy()
    # Filter out very short/empty to avoid junk vectors
    sub = sub[sub["text"].str.len() > 50].reset_index(drop=True)

    texts = sub["text"].tolist()
    sec_ids = sub["section_id"].tolist()

    print(f"Embedding {len(sub)} chunks with sentence-transformers/all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X = model.encode(texts, normalize_embeddings=True).astype("float32")

    norms = np.linalg.norm(X, axis=1)
    print(f"embedding norms → mean {norms.mean():.4f}, min {norms.min():.4f}, max {norms.max():.4f}")

    # same-section vs different-section similarity
    pairs_same, sims_same = 0, []
    pairs_diff, sims_diff = 0, []
    trials = min(200, len(sub) * 2)

    for _ in range(trials):
        i, j = random.sample(range(len(sub)), 2)
        sim = float(X[i] @ X[j])
        if sec_ids[i] and sec_ids[j] and sec_ids[i] == sec_ids[j]:
            pairs_same += 1; sims_same.append(sim)
        else:
            pairs_diff += 1; sims_diff.append(sim)

    def mean_safe(a: List[float]) -> float | None:
        return round(sum(a) / len(a), 4) if a else None

    same, diff = mean_safe(sims_same), mean_safe(sims_diff)
    print(f"same-sec avg cos: {same} (n={pairs_same})")
    print(f"diff-sec avg cos: {diff} (n={pairs_diff})")

    if same is not None and diff is not None:
        delta = round(same - diff, 4)
        print(f"Δ similarity (same - diff): {delta:.4f}")
        if delta > 0.10:
            verdict = "✅ clear semantic separation"
        elif delta > 0.05:
            verdict = "⚠️ moderate separation"
        else:
            verdict = "❌ low separation — embeddings may be too generic or chunks too overlapping"
        print(verdict)
    else:
        print("⚠️ Not enough same-section pairs to compare meaningfully.")


# ---------------------------- Main ----------------------------
def main():
    args = parse_args()
    CHUNKS_PATH: Path = args.chunks

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    rows = [json.loads(l) for l in open(CHUNKS_PATH, "r", encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)

    required_cols = {"text", "section_id", "chunk_id", "level"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"⚠️ Missing expected columns: {missing}")

    print(f"✅ Loaded {len(df)} chunks from {CHUNKS_PATH}")
    print(f"unique sections: {df['section_id'].nunique()}")
    stats_print(df)
    duplicate_report(df)

    # Small, human-readable preview
    prev = df.head(6).copy()
    prev["preview"] = prev["text"].str.replace("\n"," ", regex=False).str.slice(0, 200)
    print("\nSample rows:")
    for _, r in prev.iterrows():
        print(f"- {r['chunk_id']} | sec={r['section_id']} | lvl={r['level']} | len={len(r['text'])} | {r['preview']}…")

    print("\n--- Embedding sanity check ---")
    embed_sanity(df, sample_n=args.sample, seed=args.seed)


if __name__ == "__main__":
    main()