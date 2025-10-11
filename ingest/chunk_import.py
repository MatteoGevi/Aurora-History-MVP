# ingest/chunk.py - Reusable chunking functions
from __future__ import annotations
import hashlib, re, requests, os
import pymupdf as fitz
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============ UTILITY FUNCTIONS ============

def slug(s: str) -> str:
    """Convert title to URL-safe slug"""
    s = re.sub(r"\s+", "-", s.strip().lower())
    return re.sub(r"[^a-z0-9\-]+", "", s)[:80] or "untitled"

def norm_title(s: str | None) -> str:
    """Normalize whitespace in title"""
    if not s: return ""
    return re.sub(r"\s+", " ", s.strip())

def extract_text_pages(doc: fitz.Document, p1: int, p2: int) -> str:
    """Extract text from 1-based page range"""
    parts = []
    for p in range(p1 - 1, p2):
        parts.append(doc.load_page(p).get_text("text"))
    raw = "\n".join(parts)
    raw = re.sub(r"\r\n|\r", "\n", raw)
    raw = re.sub(r"[ \t]+\n", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()

def windows(text: str, size: int = 200_000, overlap: int = 1_000) -> List[str]:
    """Split text into overlapping windows"""
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size])
        i = max(i + size - overlap, i + 1)
    return out

def flatten(nodes: List[dict]) -> List[dict]:
    """Flatten nested ToC structure"""
    bag = []
    def walk(n):
        bag.append(n)
        for ch in n.get("children", []):
            walk(ch)
    for n in nodes:
        walk(n)
    return bag

def descendants(node: dict) -> List[dict]:
    """Get all descendant nodes"""
    out = []
    def walk(n):
        for ch in n.get("children", []):
            out.append(ch)
            walk(ch)
    walk(node)
    return out

# ============ PDF FETCHING ============

def fetch_pdf_from_storage(
    supabase_url: str, 
    service_key: str,  # Use SERVICE_ROLE key, not anon key
    bucket: str, 
    filename: str
) -> bytes:
    """
    Fetch PDF from private Supabase Storage bucket.
    Requires service_role key for backend operations.
    """
    supabase = create_client(supabase_url, service_key)
    
    try:
        # Download from private bucket
        res = supabase.storage.from_(bucket).download(filename)
        print(f"✅ Downloaded {len(res)} bytes")
        return res
    except Exception as e:
        raise RuntimeError(f"Failed to fetch PDF from private bucket: {e}")

# ============ TOC BUILDING ============

def build_toc(pdf_bytes: bytes) -> Tuple[str, List[dict], int]:
    """
    Build hierarchical ToC from PDF bytes.
    Returns: (doc_key, toc_tree, page_count)
    """
    doc_key = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        toc_raw = doc.get_toc(simple=True) or []
        last_page = doc.page_count
        
        toc, stack = [], []
        idx = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
        
        def push_node(level: int, title: str, page_start: int):
            for l in range(level, 7):
                idx[l] = idx[l] + 1 if l == level else 0
            path = "-".join(str(idx[l]) for l in range(1,7) if idx[l] > 0)
            node = {
                "id": f"h{level}-{path}__{slug(title)}",
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
        
        for lvl, title, p1 in toc_raw:
            push_node(int(lvl), norm_title(title), int(p1))
        
        while stack:
            stack[-1]["page_end"] = last_page
            stack.pop()
    
    return doc_key, toc, last_page

# ============ CHUNKING ============

def chapter_intervals(chapter: dict) -> List[Tuple[int, int, dict]]:
    """
    Return (page_start, page_end, node) intervals for all children + leftovers.
    """
    chapter_start, chapter_end = chapter["page_start"], chapter["page_end"]
    kids = descendants(chapter)

    # Get child intervals
    child_ints = []
    for k in kids:
        s = max(k["page_start"], chapter_start)
        e = min(k["page_end"], chapter_end)
        if s <= e:
            child_ints.append((s, e, k))

    # Merge overlapping intervals
    child_ints.sort(key=lambda x: (x[0], x[1]))
    merged = []
    for s, e, k in child_ints:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    # Find leftover pages
    leftovers = []
    cursor = chapter_start
    for s, e in merged:
        if cursor < s:
            leftovers.append((cursor, s-1, chapter))
        cursor = max(cursor, e + 1)
    if cursor <= chapter_end:
        leftovers.append((cursor, chapter_end, chapter))

    # Combine children + leftovers
    work = []
    seen = set()
    for k in kids:
        key = (k["id"], k["page_start"], k["page_end"])
        if key not in seen:
            seen.add(key)
            s = max(k["page_start"], chapter_start)
            e = min(k["page_end"], chapter_end)
            work.append((s, e, k))
    work.extend(leftovers)
    
    return sorted(work, key=lambda x: (x[0], x[1]))

def chunk_sections(
    pdf_bytes: bytes,
    toc: List[dict],
    chunk_size: int = 1000,
    overlap: int = 150
) -> List[dict]:
    """
    Chunk all H1 sections and their children.
    Returns list of chunk dicts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    chunks = []
    flat_nodes = flatten(toc)
    
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        seq_by_section = {}
        
        for chapter in [n for n in flat_nodes if n["level"] == 1]:
            work = chapter_intervals(chapter)
            
            for page_start, page_end, sec in work:
                if page_start > page_end:
                    continue
                
                text = extract_text_pages(doc, page_start, page_end)
                if not text:
                    continue
                
                # Window text to avoid SpaCy limits
                docs = []
                for slab in windows(text, size=200_000, overlap=1_000):
                    docs.extend(splitter.create_documents([slab]))
                
                sid = sec["id"]
                seq_by_section[sid] = seq_by_section.get(sid, 0)
                
                for d in docs:
                    chunk_text = (d.page_content or "").strip()
                    if len(chunk_text) < 100:
                        continue
                    
                    seq_by_section[sid] += 1
                    
                    chunks.append({
                        "section_id": sid,
                        "chunk_seq": seq_by_section[sid],
                        "section_title": sec["title"],
                        "level": sec["level"],
                        "page_start": page_start,
                        "page_end": page_end,
                        "text": chunk_text
                    })
    
    return chunks