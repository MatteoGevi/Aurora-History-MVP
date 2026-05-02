# ingest/chunk.py - Reusable chunking functions
from __future__ import annotations
import hashlib, re, requests, os
import pymupdf as fitz
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter, SpacyTextSplitter

# ============ PDF FETCHING ============

def fetch_pdf_from_storage(
    supabase_url: str,
    auth_token: str,
    bucket: str,
    filename: str
) -> bytes:
    """
    Fetch PDF from Supabase Storage.
    auth_token can be a service role key (bypasses RLS) or a user JWT (respects RLS).
    """
    auth_url = f"{supabase_url}/storage/v1/object/authenticated/{bucket}/{filename}"

    headers = {
        "apikey": auth_token,
        "Authorization": f"Bearer {auth_token}"
    }
    
    print(f"📥 Fetching PDF...")
    resp = requests.get(auth_url, headers=headers, timeout=30)
    resp.raise_for_status()
    
    pdf_bytes = resp.content
    if not pdf_bytes.startswith(b'%PDF'):
        raise ValueError("Downloaded file is not a valid PDF")
    
    print(f"✅ Downloaded {len(pdf_bytes):,} bytes ({len(pdf_bytes) / 1024 / 1024:.2f} MB)")
    return pdf_bytes

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

def get_ancestor_titles(node: dict, flat_nodes: List[dict]) -> List[Tuple[int, str]]:
    """Get all ancestor section titles with their levels."""
    ancestors = []
    current_level = node["level"]
    current_page = node["page_start"]
    
    # Find all ancestors by walking backwards through nodes
    for n in reversed(flat_nodes):
        if n["page_start"] <= current_page and n["level"] < current_level:
            ancestors.insert(0, (n["level"], n["title"]))
            current_level = n["level"]
            if current_level == 1:
                break
    
    return ancestors

# ============ TOC BUILDING ============

def _detect_headings_by_font(doc: fitz.Document) -> List[Tuple[int, str, int]]:
    """
    Detect headings by comparing span font sizes against the most common (body) size.
    Returns list of (level, title, page_1based) tuples, filtering out running headers.
    """
    from collections import Counter

    # Pass 1: find body font size (most frequent)
    sizes = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = span.get("size", 0)
                    if text and size > 4:
                        sizes.append(round(size, 1))

    if not sizes:
        return []

    body_size = Counter(sizes).most_common(1)[0][0]

    # Pass 2: collect candidate headings
    candidates: List[Tuple[int, str, int]] = []
    occurrences: Counter = Counter()

    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                line_text = "".join(s.get("text", "") for s in spans).strip()
                if not line_text or len(line_text) < 3 or len(line_text) > 150:
                    continue
                if line_text.isdigit():
                    continue

                max_size = max(s.get("size", 0) for s in spans)
                is_bold = any(s.get("flags", 0) & 16 for s in spans)
                ratio = max_size / body_size if body_size > 0 else 1.0

                if ratio >= 1.4:
                    level = 1
                elif ratio >= 1.2 or (ratio >= 1.1 and is_bold):
                    level = 2
                elif is_bold and ratio >= 0.95:
                    level = 3
                else:
                    continue

                occurrences[line_text] += 1
                candidates.append((level, line_text, page_num))

    # Filter running headers/footers (text appearing on > 20 % of pages)
    repeat_threshold = max(2, doc.page_count * 0.20)
    return [
        (lvl, txt, pg) for lvl, txt, pg in candidates
        if occurrences[txt] < repeat_threshold
    ]


def _detect_headings_by_pattern(doc: fitz.Document) -> List[Tuple[int, str, int]]:
    """
    Detect headings using structural text patterns (numbered sections, chapter markers).
    Returns list of (level, title, page_1based) tuples.
    """
    H1 = [
        re.compile(r'^(chapter\s+[\d ivxlcIVXLC]+[\.\:]?\s*.{0,80})$', re.IGNORECASE),
        re.compile(r'^(part\s+[\d ivxlcIVXLC]+[\.\:]?\s*.{0,80})$', re.IGNORECASE),
        re.compile(r'^(\d+\.\s+[A-ZÀ-ɏ].{2,80})$'),
    ]
    H2 = [
        re.compile(r'^(\d+\.\d+\s+[A-ZÀ-ɏ].{1,70})$'),
        re.compile(r'^(section\s+\d+[\.\:]?\s*.{0,70})$', re.IGNORECASE),
    ]
    H3 = [
        re.compile(r'^(\d+\.\d+\.\d+\s+[A-ZÀ-ɏ].{1,60})$'),
    ]

    entries: List[Tuple[int, str, int]] = []
    seen: set = set()

    for page_num, page in enumerate(doc, start=1):
        for line in page.get_text("text").split("\n"):
            line = line.strip()
            if not line or len(line) > 150:
                continue

            level = None
            for pat in H1:
                if pat.match(line):
                    level = 1
                    break
            if level is None:
                for pat in H2:
                    if pat.match(line):
                        level = 2
                        break
            if level is None:
                for pat in H3:
                    if pat.match(line):
                        level = 3
                        break

            if level is not None:
                key = (line, page_num)
                if key not in seen:
                    seen.add(key)
                    entries.append((level, line, page_num))

    return entries


def build_toc(pdf_bytes: bytes) -> Tuple[str, List[dict], int]:
    """
    Build hierarchical ToC from PDF bytes.
    First tries embedded PDF bookmarks, then font-based heading detection,
    then regex pattern detection.  Returns: (doc_key, toc_tree, page_count)
    """
    doc_key = hashlib.sha256(pdf_bytes).hexdigest()[:16]

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        toc_raw = doc.get_toc(simple=True) or []
        last_page = doc.page_count

        if not toc_raw:
            toc_raw = _detect_headings_by_font(doc)
            if len(toc_raw) >= 2:
                print("📖 No embedded ToC — built synthetic ToC via font-size analysis "
                      f"({len(toc_raw)} headings detected)")
            else:
                toc_raw = _detect_headings_by_pattern(doc)
                if len(toc_raw) >= 2:
                    print("📖 No embedded ToC — built synthetic ToC via text patterns "
                          f"({len(toc_raw)} headings detected)")
                else:
                    toc_raw = []

            # Normalize: promote the minimum detected level to 1 so chunk_sections_with_hierarchy
            # always has at least one H1 chapter to iterate over.
            if toc_raw:
                min_level = min(lvl for lvl, _, _ in toc_raw)
                if min_level > 1:
                    shift = min_level - 1
                    toc_raw = [(lvl - shift, title, page) for lvl, title, page in toc_raw]

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

def chunk_by_pages(
    pdf_bytes: bytes,
    chunk_pages: int = 3,
    overlap_pages: int = 1,
) -> List[dict]:
    """Page-based fallback chunking for PDFs with no embedded ToC."""
    chunks = []
    step = chunk_pages - overlap_pages

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_count = doc.page_count
        start = 1
        while start <= page_count:
            end = min(start + chunk_pages - 1, page_count)
            text = extract_text_pages(doc, start, end)
            if text:
                sid = f"h1-1__page-{start}-{end}"
                chunks.append({
                    "section_id": sid,
                    "chunk_seq": 1,
                    "section_title": f"Page {start}\u2013{end}",
                    "level": 1,
                    "page_start": start,
                    "page_end": end,
                    "text": text,
                })
            if end == page_count:
                break
            start += step

    return chunks


def chunk_sections_with_hierarchy(
    pdf_bytes: bytes,
    toc: List[dict],
    chunk_size: int = 2000,
    chunk_overlap: int = 50
) -> List[dict]:
    """
    Chunk all H1 sections and their children using SpaCy sentence-aware splitting
    with full hierarchical markdown headers for better semantic search.
    
    Returns list of chunk dicts with 'hierarchy' field included.
    """
    # Use SpaCy for sentence-aware splitting
    splitter = SpacyTextSplitter(
        pipeline="en_core_web_sm",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
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
                
                # Build full header hierarchy
                ancestors = get_ancestor_titles(sec, flat_nodes)
                header_parts = []
                
                # Add ancestor headers
                for level, title in ancestors:
                    header_parts.append(f"{'#' * level} {title}")
                
                # Add current section header
                header_parts.append(f"{'#' * sec['level']} {sec['title']}")
                
                header = "\n\n".join(header_parts) + "\n\n"
                
                # Window text to avoid SpaCy limits (E088 error)
                docs = []
                for slab in windows(text, size=200_000, overlap=1_000):
                    # Combine header with content for first window only
                    if not docs:
                        windowed_text = header + slab
                    else:
                        windowed_text = slab
                    
                    docs.extend(splitter.create_documents([windowed_text]))
                
                sid = sec["id"]
                seq_by_section[sid] = seq_by_section.get(sid, 0)
                
                for i, d in enumerate(docs):
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
                        "text": chunk_text,
                    })
    
    return chunks