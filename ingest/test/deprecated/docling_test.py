# docling_test.py
from docling_parse import pdf
from sentence_transformers import SentenceTransformer

doc = pdf.parse("data/textbook.pdf")         # structured parse
md  = doc.to_markdown()                      # headings like #, ##, ###

# Split on markdown headers
chunks = [part.strip() for part in md.split("\n# ") if part.strip()]
chunks = [("# " + c) if not c.startswith("# ") else c for c in chunks]

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
vecs = model.encode(chunks, normalize_embeddings=True)
print("chunks:", len(chunks), "dim:", vecs.shape[1])
print(chunks[0][:300])
