# chunk.py - Build ToC and chunks from PDF in Supabase Storage
from __future__ import annotations
from pathlib import Path
import sys, json, re, hashlib, os
import requests

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import slug, norm_title, extract_text_pages, windows, flatten, descendants
import pymupdf as fitz
from langchain_text_splitters import SpacyTextSplitter, RecursiveCharacterTextSplitter

# ---------------- CONFIG ----------------
OUT_DIR = ROOT / "data" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CHARS = 1000
OVERLAP_CHARS = 150

# Supabase config from environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET", "textbook")
PDF_FILENAME = os.getenv("PDF_FILENAME")

if not all([SUPABASE_URL, SUPABASE_ANON_KEY, PDF_FILENAME]):
    print("ERROR: Missing required env vars:")
    print("  SUPABASE_URL, SUPABASE_ANON_KEY, PDF_FILENAME")
    sys.exit(1)

# ---------------- FETCH PDF FROM SUPABASE STORAGE ----------------
def fetch_pdf_from_storage() -> bytes:
    """Fetch PDF using public URL or signed URL depending on bucket policy"""
    # Try public URL first (if bucket is public)
    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{PDF_FILENAME}"
    
    headers = {"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"}
    
    try:
        print(f"Fetching from: {public_url}")
        resp = requests.get(public_url, headers=headers, timeout=30)
        
        if resp.status_code == 404:
            # Try authenticated endpoint if public fails
            auth_url = f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{PDF_FILENAME}"
            print(f"Public fetch failed, trying authenticated: {auth_url}")
            resp = requests.get(auth_url, headers=headers, timeout=30)
        
        resp.raise_for_status()
        print(f"Downloaded {len(resp.content)} bytes")
        return resp.content
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR fetching PDF: {e}")
        sys.exit(1)

pdf_bytes = fetch_pdf_from_storage()
doc_key = hashlib.sha256(pdf_bytes).hexdigest()[:16]

# Cache locally for inspection
cache_path = OUT_DIR / f"{doc_key}.pdf"
cache_path.write_bytes(pdf_bytes)
print(f"Cached to: {cache_path}")

# ---------------- BUILD PDF TOC ----------------
with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
    toc_raw = doc.get_toc(simple=True) or []
    last_page = doc.page_count
    
    toc: list[dict] = []
    stack: list[dict] = []
    idx = {1:0,2:0,3:0,4:0,5:0,6:0}

    def push_node(level: int, title: str, page_start: int):
        for l in range(level, 7):
            if l == level: idx[l] += 1
            else: idx[l] = 0
        path = "-".join(str(idx[l]) for l in range(1,7) if idx[l] > 0)
        nid = f"h{level}-{path}__{slug(title)}"
        node = {
            "id": nid,
            "title": title,
            "level": level,
            "page_start": page_start,
            "page_end": None,
            "children": [],
        }
        while stack and stack[-1]["level"] >= level:
            stack[-1]["page_end"] = max(page_start - 1, stack[-1]["page_start"])
            stack.pop()
        if not stack:
            toc.append(node)
        else:
            stack[-1]["children"].append(node)
        stack.append(node)

    for lvl, title, p1 in toc_raw:
        push_node(int(lvl), norm_title(title), int(p1))

    while stack:
        stack[-1]["page_end"] = last_page
        stack.pop()

flat_nodes = flatten(toc)

# Save ToC
toc_file = OUT_DIR / "toc.json"
toc_file.write_text(
    json.dumps({
        "doc_key": doc_key,
        "doc_title": PDF_FILENAME,
        "page_count": last_page,
        "nodes": toc
    }, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"Wrote: {toc_file}")

# ---------------- BUILD SPLITTER ----------------
try:
    splitter = SpacyTextSplitter(
        pipeline="en_core_web_sm",
        chunk_size=TARGET_CHARS,
        chunk_overlap=OVERLAP_CHARS,
    )
except Exception as e:
    print(f"[WARN] SpaCy unavailable, using fallback: {e}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TARGET_CHARS,
        chunk_overlap=OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " "],
    )

# ---------------- CHUNK ALL SECTIONS ----------------
def chapter_intervals(chapter: dict):
    """Return (start,end,node) intervals for all children plus chapter leftovers."""
    chapter_start, chapter_end = chapter["page_start"], chapter["page_end"]
    kids = descendants(chapter)

    child_ints = []
    for k in kids:
        s, e = max(k["page_start"], chapter_start), min(k["page_end"], chapter_end)
        if s <= e:
            child_ints.append((s, e, k))

    child_ints.sort(key=lambda x: (x[0], x[1]))
    merged = []
    for s, e, k in child_ints:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    leftovers = []
    cursor = chapter_start
    for s, e in merged:
        if cursor < s:
            leftovers.append((cursor, s-1, chapter))
        cursor = max(cursor, e + 1)
    if cursor <= chapter_end:
        leftovers.append((cursor, chapter_end, chapter))

    work = []
    seen = set()
    for k in kids:
        key = (k["id"], k["page_start"], k["page_end"])
        if key not in seen:
            seen.add(key)
            work.append((max(k["page_start"], chapter_start), min(k["page_end"], chapter_end), k))
    work.extend(leftovers)
    return sorted(work, key=lambda x: (x[0], x[1]))

out_jsonl = OUT_DIR / "chunks.jsonl"
total_chunks = 0

with fitz.open(stream=pdf_bytes, filetype="pdf") as doc, \
     out_jsonl.open("w", encoding="utf-8") as f:
    
    seq_by_section: dict[str,int] = {}

    for chapter in [n for n in flat_nodes if n["level"] == 1]:
        work = chapter_intervals(chapter)
        for s, e, sec in work:
            if s > e:
                continue
            text = extract_text_pages(doc, s, e)
            if not text:
                continue

            docs = []
            for slab in windows(text, size=200_000, overlap=1_000):
                docs.extend(splitter.create_documents([slab]))

            sid = sec["id"]
            seq_by_section[sid] = seq_by_section.get(sid, 0)

            for d in docs:
                t = (d.page_content or "").strip()
                if len(t) < 100:
                    continue
                seq_by_section[sid] += 1
                total_chunks += 1

                row = {
                    "chunk_id": f"{doc_key}::{sid}::c{seq_by_section[sid]:06d}",
                    "chunk_seq": seq_by_section[sid],
                    "doc_key": doc_key,
                    "section_id": sid,
                    "section_title": sec["title"],
                    "level": sec["level"],
                    "page_start": s,
                    "page_end": e,
                    "text": t,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote: {out_jsonl}")
print(f"Total chunks: {total_chunks}")