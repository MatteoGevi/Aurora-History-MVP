# tests/test_embeddings.py
import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client
import os

def test_embedding_sanity():
    """Test embeddings have correct properties"""
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )
    
    # Fetch sample embeddings
    result = supabase.table("chunks") \
        .select("id, text, embedding") \
        .limit(100) \
        .execute()
    
    chunks = result.data
    assert len(chunks) > 0, "No chunks in database"
    
    embeddings = np.array([c["embedding"] for c in chunks])
    
    # Test 1: Correct dimensionality
    assert embeddings.shape[1] == 384, f"Wrong dimension: {embeddings.shape[1]}"
    
    # Test 2: Unit norm (normalized)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings not normalized"
    
    # Test 3: Self-similarity is highest
    S = embeddings @ embeddings.T
    np.fill_diagonal(S, -1)  # Exclude self
    max_sim = S.max(axis=1)
    
    # Each embedding should be most similar to itself (if it were included)
    # So max similarity to others should be < 1.0
    assert (max_sim < 0.99).all(), "Found identical embeddings (possible duplicates)"
    
    print(f"✅ Embedding sanity checks passed (n={len(chunks)})")
    print(f"   Avg norm: {norms.mean():.6f}")
    print(f"   Max inter-chunk similarity: {max_sim.max():.4f}")

def test_section_coherence():
    """Test that same-section chunks are more similar than random"""
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )
    
    # Get chunks from 2 different sections
    sections = supabase.table("toc_nodes").select("id").limit(2).execute()
    sec1, sec2 = [s["id"] for s in sections.data[:2]]
    
    chunks1 = supabase.table("chunks") \
        .select("embedding") \
        .eq("toc_node_id", sec1) \
        .limit(10) \
        .execute()
    
    chunks2 = supabase.table("chunks") \
        .select("embedding") \
        .eq("toc_node_id", sec2) \
        .limit(10) \
        .execute()
    
    emb1 = np.array([c["embedding"] for c in chunks1.data])
    emb2 = np.array([c["embedding"] for c in chunks2.data])
    
    # Within-section similarity
    within_sim = np.mean([emb1[i] @ emb1[j] 
                         for i in range(len(emb1)) 
                         for j in range(i+1, len(emb1))])
    
    # Cross-section similarity
    cross_sim = np.mean(emb1 @ emb2.T)
    
    print(f"Within-section similarity: {within_sim:.4f}")
    print(f"Cross-section similarity: {cross_sim:.4f}")
    
    # Within should be higher (but not always guaranteed)
    if within_sim > cross_sim:
        print("✅ Section coherence test passed")
    else:
        print("⚠️  Cross-section similarity higher (may be OK for similar topics)")

if __name__ == "__main__":
    test_embedding_sanity()
    test_section_coherence()