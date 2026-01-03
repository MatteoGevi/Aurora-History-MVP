# ingest/embed.py - Embedding utilities
from typing import List, Dict, Tuple, Optional
import numpy as np
from toc_chunk import flatten
from sentence_transformers import SentenceTransformer

def generate_embeddings(
    texts: List[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
    normalize: bool = True,
    show_progress: bool = True
) -> np.ndarray:

    print(f"🤖 Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"⚙️  Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize,
        convert_to_numpy=True  # Explicit numpy conversion
    )
    
    print(f"✅ Generated embeddings with shape: {embeddings.shape}")
    return embeddings

def validate_embeddings(
    embeddings: np.ndarray, 
    expected_dim: Optional[int] = None
) -> Tuple[bool, List[str]]:
    """
    Validate embedding quality.
    
    Returns: 
        (passed, list_of_issues)
    """
    issues = []
    
    # Check shape
    if len(embeddings.shape) != 2:
        issues.append(f"❌ Wrong shape: {embeddings.shape}, expected 2D array")
        return False, issues
    
    # Check expected dimension
    if expected_dim and embeddings.shape[1] != expected_dim:
        issues.append(f"❌ Wrong embedding dimension: {embeddings.shape[1]}, expected {expected_dim}")
    
    # Check normalization (important for cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        issues.append(f"⚠️  Not L2-normalized (mean norm: {norms.mean():.6f}, should be ~1.0)")
    
    # Check for NaN/Inf
    if np.isnan(embeddings).any():
        issues.append("❌ Contains NaN values")
    if np.isinf(embeddings).any():
        issues.append("❌ Contains Inf values")
    
    # Check for all-zero vectors
    zero_vectors = np.all(embeddings == 0, axis=1).sum()
    if zero_vectors > 0:
        issues.append(f"⚠️  Found {zero_vectors} all-zero embedding vectors")
    
    # Check for near-duplicates (might indicate data quality issues)
    if len(embeddings) > 1:
        S = embeddings @ embeddings.T
        np.fill_diagonal(S, 0)
        exact_matches = np.sum(S > 0.9999)
        if exact_matches > 0:
            issues.append(f"⚠️  Found {exact_matches} near-duplicate embeddings (similarity > 0.9999)")
    
    passed = len(issues) == 0
    return passed, issues

def ingest_to_supabase(
    doc_key: str,
    toc: List[dict],
    chunks: List[dict],
    embeddings: np.ndarray,
    supabase_client
):
    """
    Insert ToC nodes and chunks with embeddings into Supabase.
    
    Args:
        doc_key: Unique document identifier
        toc: Table of contents tree structure
        chunks: List of chunk dictionaries (should include 'hierarchy' field)
        embeddings: numpy array of embeddings (must match len(chunks))
        supabase_client: Supabase client instance
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"Chunk count ({len(chunks)}) doesn't match embedding count ({len(embeddings)})")
    
    # 1. Insert ToC nodes
    print("📚 Inserting ToC nodes...")
    toc_flat = flatten(toc)
    toc_records = [{
        "node_id": n["id"],
        "title": n["title"],
        "level": n["level"],
        "page_start": n["page_start"],
        "page_end": n["page_end"]
    } for n in toc_flat]
    
    result = supabase_client.table("toc_nodes").upsert(
        toc_records, 
        on_conflict="node_id"
    ).execute()
    print(f"✅ Inserted {len(toc_records)} ToC nodes")
    
    # 2. Get node_id -> db_id mapping
    print("🔍 Fetching node ID mappings...")
    result = supabase_client.table("toc_nodes").select("id, node_id").execute()
    node_map = {r["node_id"]: r["id"] for r in result.data}
    
    # 3. Prepare chunks with embeddings
    print(f"\n📦 Preparing {len(chunks)} chunks with embeddings...")
    chunk_records = []
    for i, c in enumerate(chunks):
        # Build chunk record
        chunk_record = {
            "chunk_id": f"{doc_key}::{c['section_id']}::c{c['chunk_seq']:06d}",
            "toc_node_id": node_map.get(c["section_id"]),
            "chunk_seq": c["chunk_seq"],
            "section_title": c["section_title"],
            "level": c["level"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "text": c["text"],
            "embedding": embeddings[i].tolist()  # Convert numpy to list for JSON
        }
        
        chunk_records.append(chunk_record)
    
    # 4. Insert chunks in batches (Supabase has payload limits)
    print(f"\n💾 Inserting chunks...")
    BATCH_SIZE = 100
    total_batches = (len(chunk_records) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(chunk_records), BATCH_SIZE):
        batch = chunk_records[i:i+BATCH_SIZE]
        supabase_client.table("chunks").upsert(
            batch,
            on_conflict="chunk_id"
        ).execute()
        batch_num = i // BATCH_SIZE + 1
        print(f"   ✓ Batch {batch_num}/{total_batches} ({len(batch)} chunks)")
    
    print(f"\n✅ Successfully inserted {len(chunk_records)} chunks with embeddings")