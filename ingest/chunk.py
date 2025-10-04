from __future__ import annotations
from pathlib import Path
import sys, json, re, hashlib
import fitz
from langchain_text_splitters import SpacyTextSplitter, RecursiveCharacterTextSplitter

# ---------------- CONFIG ----------------
ROOT = Path(__file__).resolve().parents[1]
PDF_NAME = "AI Engineering.pdf"
PDF_PATH = ROOT / "data" / PDF_NAME
OUT_DIR = ROOT / "data" / "out"
TARGET_CHARS = 1000
OVERLAP_CHARS = 150

if len(sys.argv) > 1:
    PDF_PATH = Path(sys.argv[1]).expanduser().resolve()
if len(sys.argv) > 2:
    OUT_DIR = Path(sys.argv[2]).expanduser().resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: silence noisy MuPDF repair logs
try:
    fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass

# ---------------- HELPERS ----------------
def slug(s: str) -> str:
    s = re.sub(r"\s+", "-", s.strip().lower())
    return re.sub(r"[^a-z0-9\-]+", "", s)[:80] or "untitled"

def norm_title(s: str|None) -> str:
    if not s: return ""
    return re.sub(r"\s+", " ", s.strip())

def read_doc_key(pdf_path: Path) -> str:
    try:
        data = pdf_path.read_bytes()
        return hashlib.sha256(data).hexdigest()[:16]
    except Exception:
        return pdf_path.name  # fallback

def extract_text_pages(doc: fitz.Document, p1: int, p2: int) -> str:
    """Inclusive 1-based page range → cleaned text."""
    parts = []
    for p in range(p1 - 1, p2):
        parts.append(doc.load_page(p).get_text("text"))
    raw = "\n".join(parts)
    raw = re.sub(r"\r\n|\r", "\n", raw)
    raw = re.sub(r"[ \t]+\n", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()

def windows(text: str, size: int = 200_000, overlap: int = 1_000) -> list[str]:
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size])
        i = max(i + size - overlap, i + 1)
    return out

# ---------------- BUILD PDF TOC ----------------
print("Using:", PDF_PATH)
with fitz.open(str(PDF_PATH)) as doc:
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
    return node

for lvl, title, p1 in toc_raw:
    push_node(int(lvl), norm_title(title), int(p1))

while stack:
    stack[-1]["page_end"] = last_page
    stack.pop()

def flatten(nodes):
    bag = []
    def walk(n):
        bag.append(n)
        for ch in n.get("children", []):
            walk(ch)
    for n in nodes:
        walk(n)
    return bag

def descendants(node):
    out = []
    def walk(n):
        for ch in n.get("children", []):
            out.append(ch)
            walk(ch)
    walk(node)
    return out

flat_nodes = flatten(toc)

# Save ToC (for later inspection/debugging)
OUT_DIR.joinpath("toc.json").write_text(
    json.dumps({"doc_title": PDF_PATH.name, "nodes": toc}, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print("Wrote:", OUT_DIR / "toc.json")

# ---------------- BUILD SPLITTER ----------------
# Use model-name path for compatibility (0.3.11 expects string)
try:
    splitter = SpacyTextSplitter(
        pipeline="en_core_web_sm",  # ensure model installed: poetry run python -m spacy download en_core_web_sm
        chunk_size=TARGET_CHARS,
        chunk_overlap=OVERLAP_CHARS,
    )
except Exception as e:
    print("[WARN] SpacyTextSplitter unavailable; falling back. Error:", e)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TARGET_CHARS,
        chunk_overlap=OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " "],
    )

# ---------------- CHUNK ALL CHAPTERS (H1 + children + leftovers) ----------------
doc_key = read_doc_key(PDF_PATH)
out_jsonl = OUT_DIR / "chunks.paragraphs.jsonl"

def chapter_intervals(chapter: dict):
    """Return (start,end,node) intervals for all children plus chapter leftovers."""
    chapter_start, chapter_end = chapter["page_start"], chapter["page_end"]
    kids = descendants(chapter)

    # sanitize and clamp children to chapter bounds
    child_ints = []
    for k in kids:
        s, e = max(k["page_start"], chapter_start), min(k["page_end"], chapter_end)
        if s <= e:
            child_ints.append((s, e, k))

    # merge to get covered regions (for computing leftovers)
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

    # final worklist: each child interval by its own node, plus leftovers (attributed to H1)
    work = []
    seen = set()
    for k in kids:
        key = (k["id"], k["page_start"], k["page_end"])
        if key not in seen:
            seen.add(key)
            work.append((max(k["page_start"], chapter_start), min(k["page_end"], chapter_end), k))
    work.extend(leftovers)
    return sorted(work, key=lambda x: (x[0], x[1]))

total_chunks = 0
with fitz.open(str(PDF_PATH)) as doc, out_jsonl.open("w", encoding="utf-8") as f:
    seq_by_section: dict[str,int] = {}

    # iterate every H1 chapter
    for chapter in [n for n in flat_nodes if n["level"] == 1]:
        work = chapter_intervals(chapter)
        for s, e, sec in work:
            if s > e:
                continue
            text = extract_text_pages(doc, s, e)
            if not text:
                continue

            # pre-window to avoid spaCy E088
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
                    "page_range": [s, e],
                    "text": t,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Wrote:", out_jsonl)
print("Done. Total chunks:", total_chunks)