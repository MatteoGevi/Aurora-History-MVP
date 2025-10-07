import os, sys, json, time, argparse, hashlib, pathlib, re, pymupdf as fitz
from typing import List, Dict
from pathlib import Path

# Slugify a string
def slug(s: str) -> str:
    s = re.sub(r"\s+", "-", s.strip().lower())
    return re.sub(r"[^a-z0-9\-]+", "", s)[:80] or "untitled"

# Normalize a title
def norm_title(s: str|None) -> str:
    if not s: return ""
    return re.sub(r"\s+", " ", s.strip())

# Read a document key
def read_doc_key(pdf_path: Path) -> str:
    try:
        data = pdf_path.read_bytes()
        return hashlib.sha256(data).hexdigest()[:16]
    except Exception:
        return pdf_path.name  # fallback

# Extract text from a document
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

# Window text
def windows(text: str, size: int = 200_000, overlap: int = 1_000) -> list[str]:
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size])
        i = max(i + size - overlap, i + 1)
    return out

# Flatten nodes
def flatten(nodes):
    bag = []
    def walk(n):
        bag.append(n)
        for ch in n.get("children", []):
            walk(ch)
    for n in nodes:
        walk(n)
    return bag

# Descendants
def descendants(node):
    out = []
    def walk(n):
        for ch in n.get("children", []):
            out.append(ch)
            walk(ch)
    walk(node)
    return out