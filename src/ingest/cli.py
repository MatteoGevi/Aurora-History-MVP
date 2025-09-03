# src/ingest/cli.py
import json, os, sys
from .pdf_extract import extract_with_pymupdf
from .headings import detect_headings
from .toc_builder import build_toc
from .chunk import make_chunks

ART_DIR = "artifacts"

def main(pdf_path: str):
    os.makedirs(ART_DIR, exist_ok=True)
    doc = extract_with_pymupdf(pdf_path)
    heading_candidates = detect_headings(doc.blocks)
    toc_nodes = build_toc(doc.bookmarks, heading_candidates)

    # persist toc
    with open(os.path.join(ART_DIR, "toc.jsonl"), "w", encoding="utf-8") as f:
        for n in toc_nodes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")

    chunks = make_chunks(doc.blocks, toc_nodes, token_target=900, overlap_sents=2)
    with open(os.path.join(ART_DIR, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[ok] ToC nodes: {len(toc_nodes)}  → artifacts/toc.jsonl")
    print(f"[ok] Chunks:    {len(chunks)}      → artifacts/chunks.jsonl")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest.cli data/textbook.pdf"); sys.exit(1)
    main(sys.argv[1])
