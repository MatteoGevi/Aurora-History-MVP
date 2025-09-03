# ingest_test.py
from unstructured.partition.pdf import partition_pdf
from sentence_transformers import SentenceTransformer
from itertools import islice

elements = partition_pdf("data/textbook.pdf", strategy="hi_res")  # headings preserved
# Keep only Titles + body text; drop headers/footers if present
keep = [e for e in elements if e.category in ("Title","NarrativeText","ListItem","Table")]
# Simple header-aware chunking: start a new chunk on every Title
chunks, cur = [], []
for e in keep:
    if e.category == "Title" and cur:
        chunks.append("\n".join(cur)); cur = []
    cur.append(e.text.strip())
if cur: chunks.append("\n".join(cur))

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
vecs = model.encode(chunks, normalize_embeddings=True)

print(f"Chunks: {len(chunks)}  Dim: {vecs.shape[1]}")
for i, (c, v) in enumerate(islice(zip(chunks, vecs), 3)):
    print("---")
    print(c[:280].replace("\n"," "))
    print("‣ L2 norm:", (v**2).sum()**0.5)
