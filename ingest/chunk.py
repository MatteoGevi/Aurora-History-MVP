# src/ingest/test/pymupdf_test.py
from __future__ import annotations
from pathlib import Path
import sys, json, re, uuid, hashlib
import fitz
import pymupdf4llm

# NEW: add splitters
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    SpacyTextSplitter,
    RecursiveCharacterTextSplitter,
)

ROOT = Path(__file__).resolve().parents[1]
PDF_NAME = "AI Engineering.pdf"
PDF_PATH = ROOT / "data" / PDF_NAME
OUT_DIR = ROOT / "data" / "out"

if len(sys.argv) > 1:
    PDF_PATH = Path(sys.argv[1]).expanduser().resolve()
if len(sys.argv) > 2:
    OUT_DIR = Path(sys.argv[2]).expanduser().resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

def slug(s: str) -> str:
    s = re.sub(r"\s+", "-", s.strip().lower())
    return re.sub(r"[^a-z0-9\-]+", "", s)[:80] or "untitled"

def norm_title(s: str|None) -> str:
    if not s: return ""
    return re.sub(r"\s+", " ", s.strip())

def read_doc_key(pdf_path: Path) -> str:
    # Prefer a stable content hash so chunk_ids are reproducible across renames
    try:
        data = pdf_path.read_bytes()
        return hashlib.sha256(data).hexdigest()[:16]
    except Exception:
        return pdf_path.name  # fallback

print("Using:", PDF_PATH)

# ---------- Build page-aware ToC from native PDF ToC ----------
with fitz.open(str(PDF_PATH)) as doc:
    toc_raw = doc.get_toc(simple=True) or []   # list of [level, title, page1based]
    last_page = doc.page_count

# Walk native ToC; close prior nodes when a same/higher level starts
toc: list[dict] = []
stack: list[dict] = []
idx = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}

def push_node(level: int, title: str, page_start: int):
    # reset deeper counters
    for l in range(level, 7):
        if l == level: idx[l] += 1
        else: idx[l] = 0
    path = "-".join(str(idx[l]) for l in range(1,7) if idx[l] > 0)
    nid = f"h{level}-{path}__{slug(title)}"
    node = {
        "id": nid,
        "title": title,
        "level": level,
        "page_start": page_start,   # 1-based
        "page_end": None,
        "children": [],
    }
    # close higher/equal levels
    while stack and stack[-1]["level"] >= level:
        stack[-1]["page_end"] = max(page_start - 1, stack[-1]["page_start"])
        stack.pop()
    if not stack:
        toc.append(node)
    else:
        stack[-1]["children"].append(node)
    stack.append(node)
    return node

for lvl, title, p1 in toc_raw:
    push_node(int(lvl), norm_title(title), int(p1))

# close remaining
while stack:
    stack[-1]["page_end"] = last_page
    stack.pop()

# Flatten ToC
def collect_nodes(nodes, bag):
    for n in nodes:
        bag.append(n)
        collect_nodes(n["children"], bag)

flat_nodes = []
collect_nodes(toc, flat_nodes)

# Deepest-first mapping title -> id (helps when h2 & h3 share similar titles)
title_to_id = {}
for n in sorted(flat_nodes, key=lambda x: x["level"], reverse=True):
    key = norm_title(n["title"])
    title_to_id.setdefault(key, n["id"])

# Fast lookups
id_to_level = {n["id"]: n["level"] for n in flat_nodes}

# Save ToC
toc_path = OUT_DIR / "toc.json"
toc_payload = {"doc_title": PDF_PATH.name, "nodes": toc}
toc_path.write_text(json.dumps(toc_payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote:", toc_path)

# ---------- Convert PDF -> Markdown and split headers ----------
md = pymupdf4llm.to_markdown(str(PDF_PATH))
header_splitter = MarkdownHeaderTextSplitter([("#","h1"),("##","h2"),("###","h3")])
nodes = header_splitter.split_text(md)
print("Markdown nodes:", len(nodes))

# ---------- Helpers for section mapping ----------
def deepest_section_id(meta) -> str|None:
    # Try h3 -> h2 -> h1 title matches against native ToC titles
    for k in ("header_h3", "header_h2", "header_h1"):
        t = norm_title(meta.get(k))
        if t and t in title_to_id:
            return title_to_id[t]
    return None

# ---------- Sentence-aware chunking with spaCy ----------
# Target ~1000 chars per chunk with ~150 overlap (characters)
TARGET_CHARS = 1000
OVERLAP_CHARS = 150

def build_spacy_splitter():
    """
    Build a SpacyTextSplitter if spaCy model is present; otherwise
    fall back to a robust RecursiveCharacterTextSplitter.
    """
    try:
        # You must have run: python -m spacy download en_core_web_sm
        return SpacyTextSplitter(
            pipeline="en_core_web_sm",
            chunk_size=TARGET_CHARS,
            chunk_overlap=OVERLAP_CHARS
        )
    except Exception as e:
        print("[WARN] spaCy not available, using RecursiveCharacterTextSplitter. Error:", e)
        return RecursiveCharacterTextSplitter(
            chunk_size=TARGET_CHARS,
            chunk_overlap=OVERLAP_CHARS,
            separators=["\n\n", "\n", ". ", " "]
        )

section_splitter = build_spacy_splitter()

# Stable doc key for chunk_id
doc_key = read_doc_key(PDF_PATH)

out_jsonl = OUT_DIR / "chunks.paragraphs.jsonl"
seq_by_section: dict[str,int] = {}  # reset counter per section_id

with out_jsonl.open("w", encoding="utf-8") as f:
    for n in nodes:
        section_id = deepest_section_id(n.metadata)  # may be None

        # Build a readable level_path (still useful for prompting)
        level_path = " > ".join(
            [t for t in (
                norm_title(n.metadata.get("header_h1")),
                norm_title(n.metadata.get("header_h2")),
                norm_title(n.metadata.get("header_h3"))
            ) if t]
        )

        # Sequence starts at 1 per section_id; if None, bucket under 'root'
        sid = section_id or "root"
        seq_by_section[sid] = seq_by_section.get(sid, 0)

        # --- NEW: sentence-aware chunking inside each Markdown header node ---
        # section_splitter.create_documents returns List[Document] with .page_content
        docs = section_splitter.create_documents([n.page_content])
        for d in docs:
            text = d.page_content.strip()
            if not text:
                continue
            # Tiny-chunk guard: merge ultra-short leftovers (rare with Spacy splitter)
            if len(text) < 100:
                continue

            seq_by_section[sid] += 1
            chunk_seq = seq_by_section[sid]
            chunk_id = f"{doc_key}::{sid}::c{chunk_seq:06d}"

            row = {
                # REQUIRED identity fields
                "chunk_id":   chunk_id,           # e.g. "ccad82e1fa0d1a9c::h3-...::c000001"
                "chunk_seq":  chunk_seq,          # per-section sequence
                "level":      id_to_level.get(section_id) if section_id else None,

                # TEXT payload
                "text":       text,

                # METADATA
                "doc_key":    doc_key,
                "level_path": level_path,
                "section_node_id": section_id,    # add explicit section id for retrieval filters
                "version":    2                   # bumped due to new chunking method
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Wrote:", out_jsonl)
print("Done.")