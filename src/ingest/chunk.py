# src/ingest/test/pymupdf_test.py
from __future__ import annotations
from pathlib import Path
import sys, json, re, uuid
import fitz
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter

ROOT = Path(__file__).resolve().parents[1]
PDF_NAME = "These truths.pdf"
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

# Build a lookup: normalized title -> id (prefer deepest first to avoid collisions)
def collect_nodes(nodes, bag):
    for n in nodes:
        bag.append(n)
        collect_nodes(n["children"], bag)
flat_nodes = []
collect_nodes(toc, flat_nodes)
# Deepest-first mapping helps when h2 & h3 share similar titles
title_to_id = {}
for n in sorted(flat_nodes, key=lambda x: x["level"], reverse=True):
    key = norm_title(n["title"])
    title_to_id.setdefault(key, n["id"])

toc_path = OUT_DIR / "toc.json"
toc_payload = {"doc_title": PDF_PATH.name, "nodes": toc}
toc_path.write_text(json.dumps(toc_payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote:", toc_path)

# ---------- Convert PDF -> Markdown and split headers ----------
md = pymupdf4llm.to_markdown(str(PDF_PATH))
splitter = MarkdownHeaderTextSplitter([("#","h1"),("##","h2"),("###","h3")])
nodes = splitter.split_text(md)
print("Markdown nodes:", len(nodes))

# ---------- Paragraph chunks linked to ToC ids ----------
def deepest_section_id(meta) -> str|None:
    # Try h3 -> h2 -> h1 title matches against native ToC titles
    for k in ("header_h3", "header_h2", "header_h1"):
        t = norm_title(meta.get(k))
        if t and t in title_to_id:
            return title_to_id[t]
    return None

def split_paras(text: str):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

out_jsonl = OUT_DIR / "chunks.paragraphs.jsonl"
with out_jsonl.open("w", encoding="utf-8") as f:
    for n in nodes:
        section_id = deepest_section_id(n.metadata)
        level_path = " > ".join([norm_title(n.metadata.get("header_h1")),
                                 norm_title(n.metadata.get("header_h2")),
                                 norm_title(n.metadata.get("header_h3"))]).strip(" >")
        for para in split_paras(n.page_content):
            row = {
                "chunk_type": "paragraph",
                "section_node_id": section_id,     # anchor to ToC; may be None if no match
                "level_path": level_path,
                "header_h1": norm_title(n.metadata.get("header_h1")),
                "header_h2": norm_title(n.metadata.get("header_h2")),
                "header_h3": norm_title(n.metadata.get("header_h3")),
                "content": para,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Wrote:", out_jsonl)
print("Done.")