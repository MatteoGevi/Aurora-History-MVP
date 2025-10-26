# src/ingest/test/pymupdf_test.py
from pathlib import Path
import sys
import fitz
from langchain_text_splitters import MarkdownHeaderTextSplitter
import pymupdf4llm


ROOT = Path(__file__).resolve().parents[3]
PDF_NAME = "These truths.pdf"
PDF_PATH = ROOT / "data" / PDF_NAME

# Optional: allow overriding via CLI arg
if len(sys.argv) > 1:
    PDF_PATH = Path(sys.argv[1]).expanduser().resolve()

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found at: {PDF_PATH}\n"
                            f"Tip: ensure the name is exact. Current data dir:\n{ROOT/'data'}")

print(f"Using PDF: {PDF_PATH}")

# Open with PyMuPDF
doc = fitz.open(str(PDF_PATH))
toc = doc.get_toc(simple=True)
print("TOC (first 5):", toc[:5])

# Convert to Markdown with headings inferred
md = pymupdf4llm.to_markdown(str(PDF_PATH))

# Split by Markdown headers to get hierarchical chunks
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
)
chunks = splitter.split_text(md)

print(f"chunks: {len(chunks)}")
for d in chunks[:10]:
    print(d.metadata.get("header_h1"), ">", d.metadata.get("header_h2"))
    print(d.page_content[:300], "\n---")