# src/ingest/embed.py
import os, json
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from typing import List, Dict
from sentence_transformers import SentenceTransformer

PG_DSN = os.getenv("PG_DSN")

def load_chunks_from_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def upsert_chunks(conn, material_id: str, toc_map: Dict[str, str], chunks: List[Dict]):
    """
    toc_map: optional mapping {toc_title -> toc_node_id}
    """
    with conn.cursor() as cur:
        for c in chunks:
            cur.execute("""
                insert into chunks(material_id, toc_node_id, page_start, page_end, header_path, text)
                values (%s, %s, %s, %s, %s, %s)
                returning id
            """, (
                material_id,
                toc_map.get(c["toc_title"]),
                c["page_start"], c["page_end"],
                c.get("header_path"),
                c["text"]
            ))
    conn.commit()

def write_embeddings(conn, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64):
    model = SentenceTransformer(model_name)
    with conn.cursor() as cur:
        cur.execute("select id, text from chunks where embedding is null limit 5000;")
        rows = cur.fetchall()
    if not rows: 
        print("[embed] nothing to embed"); return

    ids, texts = zip(*rows)
    vecs = model.encode(list(texts), batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    with conn.cursor() as cur:
        for _id, v in zip(ids, vecs):
            cur.execute("update chunks set embedding = %s where id = %s;", (list(v), _id))
    conn.commit()
    print(f"[embed] wrote {len(ids)} embeddings")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-id", required=True)
    parser.add_argument("--toc-jsonl", default="artifacts/toc.jsonl")
    parser.add_argument("--chunks-jsonl", default="artifacts/chunks.jsonl")
    args = parser.parse_args()

    conn = psycopg2.connect(PG_DSN)

    # (Optional) create a minimal ToC map; in MVP you can leave toc_node_id null
    toc_map = {}
    if os.path.exists(args.toc_jsonl):
        with open(args.toc_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                node = json.loads(line)
                # After you insert toc_nodes into DB, map title->id here
                # toc_map[node["title"]] = <id_from_db>
                pass

    chunks = load_chunks_from_jsonl(args.chunks_jsonl)
    upsert_chunks(conn, args.material_id, toc_map, chunks)
    write_embeddings(conn)