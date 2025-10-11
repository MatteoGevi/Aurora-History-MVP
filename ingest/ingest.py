# ingest.py - Fetch PDF, build ToC, chunk, embed, and insert into Supabase
from __future__ import annotations
import os, json, hashlib, requests
from pathlib import Path
from supabase import create_client, Client

# Chunking procedures
import pymupdf as fitz
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from constants import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, STORAGE_BUCKET, PDF_FILENAME

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ---------------- FETCH PDF ----------------
def fetch_pdf() -> bytes:
    url = f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{PDF_FILENAME}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content

# ---------------- BUILD TOC ----------------
def build_toc(pdf_bytes: bytes) -> tuple[str, list[dict], int]:
    doc_key = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        toc_raw = doc.get_toc(simple=True) or []
        last_page = doc.page_count
        
        toc, stack = [], []
        idx = {1:0,2:0,3:0,4:0,5:0,6:0}
        
        def push_node(level: int, title: str, page_start: int):
            for l in range(level, 7):
                idx[l] = idx[l] + 1 if l == level else 0
            path = "-".join(str(idx[l]) for l in range(1,7) if idx[l] > 0)
            node = {
                "id": f"h{level}-{path}__{slugify(title)}",
                "title": title,
                "level": level,
                "page_start": page_start,
                "page_end": None,
                "children": []
            }
            while stack and stack[-1]["level"] >= level:
                stack[-1]["page_end"] = max(page_start - 1, stack[-1]["page_start"])
                stack.pop()
            (toc if not stack else stack[-1]["children"]).append(node)
            stack.append(node)
        
        for lvl, title, p in toc_raw:
            push_node(int(lvl), title.strip(), int(p))
        
        while stack:
            stack[-1]["page_end"] = last_page
            stack.pop()
    
    return doc_key, toc, last_page

# ---------------- CHUNK TEXT ----------------
def chunk_all_sections(pdf_bytes: bytes, toc: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    chunks = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for node in flatten_toc(toc):
            if node["level"] > 3:  # Skip deep nesting
                continue
                
            text = extract_pages(doc, node["page_start"], node["page_end"])
            if not text:
                continue
            
            docs = splitter.create_documents([text])
            for i, d in enumerate(docs, 1):
                if len(d.page_content.strip()) < 100:
                    continue
                chunks.append({
                    "node_id": node["id"],
                    "title": node["title"],
                    "level": node["level"],
                    "page_start": node["page_start"],
                    "page_end": node["page_end"],
                    "chunk_seq": i,
                    "text": d.page_content.strip()
                })
    return chunks

# ---------------- INSERT INTO SUPABASE ----------------
def ingest_to_supabase(doc_key: str, toc: list[dict], chunks: list[dict]):
    # 1. Insert ToC nodes
    toc_flat = flatten_toc(toc)
    toc_records = [{
        "node_id": n["id"],
        "title": n["title"],
        "level": n["level"],
        "page_start": n["page_start"],
        "page_end": n["page_end"]
    } for n in toc_flat]
    
    supabase.table("toc_nodes").upsert(toc_records, on_conflict="node_id").execute()
    print(f"Inserted {len(toc_records)} ToC nodes")
    
    # 2. Get node_id -> db_id mapping
    result = supabase.table("toc_nodes").select("id, node_id").execute()
    node_map = {r["node_id"]: r["id"] for r in result.data}
    
    # 3. Embed chunks
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    # 4. Insert chunks with embeddings
    chunk_records = []
    for c, emb in zip(chunks, embeddings):
        chunk_records.append({
            "chunk_id": f"{doc_key}::{c['node_id']}::c{c['chunk_seq']:06d}",
            "toc_node_id": node_map.get(c["node_id"]),
            "chunk_seq": c["chunk_seq"],
            "text": c["text"],
            "embedding": emb.tolist()
        })
    
    # Insert in batches of 100
    for i in range(0, len(chunk_records), 100):
        batch = chunk_records[i:i+100]
        supabase.table("chunks").upsert(batch, on_conflict="chunk_id").execute()
        print(f"Inserted batch {i//100 + 1}")
    
    print(f"Done! Total chunks: {len(chunk_records)}")

# ---------------- HELPERS ----------------
def slugify(s: str) -> str:
    import re
    s = re.sub(r'\s+', '-', s.lower().strip())
    return re.sub(r'[^a-z0-9-]', '', s)[:80]

def flatten_toc(nodes: list[dict]) -> list[dict]:
    result = []
    def walk(n):
        result.append(n)
        for c in n.get("children", []):
            walk(c)
    for n in nodes:
        walk(n)
    return result

def extract_pages(doc, start: int, end: int) -> str:
    parts = [doc.load_page(p-1).get_text("text") for p in range(start, end+1)]
    return "\n".join(parts).strip()

if __name__ == "__main__":
    print("Fetching PDF...")
    pdf_bytes = fetch_pdf()
    
    print("Building ToC...")
    doc_key, toc, pages = build_toc(pdf_bytes)
    
    print(f"Chunking {pages} pages...")
    chunks = chunk_all_sections(pdf_bytes, toc)
    
    print("Ingesting to Supabase...")
    ingest_to_supabase(doc_key, toc, chunks)