# ingest/embed.py - Embedding utilities
from typing import List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

_model_cache: dict = {}

def generate_embeddings(
    texts: List[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
    normalize: bool = True,
    show_progress: bool = True
) -> np.ndarray:

    if model_name not in _model_cache:
        print(f"🤖 Loading model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
    model = _model_cache[model_name]

    print(f"⚙️  Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize,
        convert_to_numpy=True
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
