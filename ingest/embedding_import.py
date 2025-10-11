# ingest/embed.py - Embedding utilities
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_embeddings(
    texts: List[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    Returns: numpy array of shape (len(texts), embedding_dim)
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize
    )
    return embeddings

def validate_embeddings(embeddings: np.ndarray) -> tuple[bool, List[str]]:
    """
    Validate embedding quality.
    Returns: (passed, list_of_issues)
    """
    issues = []
    
    # Check shape
    if len(embeddings.shape) != 2:
        issues.append(f"Wrong shape: {embeddings.shape}")
        return False, issues
    
    # Check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        issues.append(f"Not normalized (mean norm: {norms.mean():.6f})")
    
    # Check for NaN/Inf
    if np.isnan(embeddings).any():
        issues.append("Contains NaN values")
    if np.isinf(embeddings).any():
        issues.append("Contains Inf values")
    
    # Check for duplicates
    S = embeddings @ embeddings.T
    np.fill_diagonal(S, 0)
    exact_matches = np.sum(S > 0.9999)
    if exact_matches > 0:
        issues.append(f"Found {exact_matches} near-duplicate embeddings")
    
    passed = len(issues) == 0
    return passed, issues