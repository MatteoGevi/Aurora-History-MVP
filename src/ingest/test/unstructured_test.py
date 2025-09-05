from pathlib import Path
import sys

# --- paths ---
ROOT = Path(__file__).resolve().parents[3]   # .../Aurora-History-MVP
PDF_NAME = "These truths.pdf"
PDF_PATH = ROOT / "data" / PDF_NAME
if len(sys.argv) > 1:
    PDF_PATH = Path(sys.argv[1]).expanduser().resolve()

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found at: {PDF_PATH}\n"
                            f"Tip: use CLI arg or rename PDF_NAME. Data dir: {ROOT/'data'}")

print(f"Using PDF: {PDF_PATH}")

# --- unstructured partition ---
from unstructured.partition.pdf import partition_pdf

# hi_res gives better layout & Title detection; falls back to OCR if needed
elements = partition_pdf(str(PDF_PATH), strategy="hi_res")

# Keep only Titles + body-like text; include lists/tables if you want them chunked under headings
keep_categories = {"Title", "NarrativeText", "ListItem", "Table"}
keep = [e for e in elements if getattr(e, "category", None) in keep_categories]

# Header-aware chunking: start a new chunk at every Title
chunks, cur = [], []
for e in keep:
    txt = (e.text or "").strip()
    if not txt:
        continue
    if e.category == "Title" and cur:
        chunks.append("\n".join(cur))
        cur = [txt]
    else:
        cur.append(txt)
if cur:
    chunks.append("\n".join(cur))

print(f"Chunks: {len(chunks)}")
for i, c in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i} (preview) ---")
    print(c[:280].replace("\n", " "))

# --- OPTIONAL: quick embedding check (can be skipped if not installed) ---
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vecs = model.encode(chunks, normalize_embeddings=True)
    print(f"\nEmbeddings: OK  shape={vecs.shape}")
    # sanity: norms ~1.0 if normalized
    import numpy as np
    norms = np.linalg.norm(vecs[:3], axis=1)
    print("Sample norms:", [round(float(n), 4) for n in norms])
except Exception as e:
    print("\n(Embeddings step skipped or failed)")
    print("Reason:", e)
    print("Tip: poetry add sentence-transformers && accelerate torch --extra-index-url https://download.pytorch.org/whl/cpu")