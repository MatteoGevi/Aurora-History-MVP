# src/ingest/pdf_extract.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import fitz  # PyMuPDF
import re

@dataclass
class PageBlock:
    page: int
    text: str
    spans: List[Dict[str, Any]]  # [{text, size, bold, font, bbox}]

@dataclass
class DocumentExtraction:
    blocks: List[PageBlock]
    bookmarks: List[Dict[str, Any]]  # [{title, page, level}]

def _span_info(span):
    text = span.get("text","")
    size = span.get("size", 0.0)
    font = span.get("font","")
    flags = span.get("flags",0)  # bold/italic bits
    bold = bool(flags & 2) or ("Bold" in font)
    return {"text": text, "size": size, "bold": bold, "font": font, "bbox": span.get("bbox")}

def extract_with_pymupdf(pdf_path: str) -> DocumentExtraction:
    doc = fitz.open(pdf_path)
    blocks: List[PageBlock] = []
    for i, page in enumerate(doc):
        rich = page.get_text("dict")
        spans_agg = []
        lines_out = []
        for block in rich.get("blocks", []):
            for line in block.get("lines", []):
                line_text = "".join([span.get("text","") for span in line.get("spans",[])])
                if line_text.strip():
                    lines_out.append(line_text)
                for span in line.get("spans", []):
                    spans_agg.append(_span_info(span))
        blocks.append(PageBlock(page=i+1, text="\n".join(lines_out), spans=spans_agg))

    # bookmarks (table of contents if present)
    toc = doc.get_toc(simple=False)  # list of [level, title, page, …]
    bookmarks = []
    for item in toc:
        level, title, page = item[:3]
        # PyMuPDF pages are 1-based in get_toc; align with our pages
        bookmarks.append({"level": level, "title": title.strip(), "page": int(page)})

    return DocumentExtraction(blocks=blocks, bookmarks=bookmarks)