# ingest/embedding_import.py - Embedding utilities (OpenAI text-embedding-3-small)
from typing import List, Tuple, Optional
import numpy as np
from openai import OpenAI

from config.constants import OPENAI_API_KEY

_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def generate_embeddings(
    texts: List[str],
    model_name: str = "text-embedding-3-small",
    dimensions: int = 384,
    batch_size: int = 64,
    normalize: bool = True,
    show_progress: bool = True,
    **_kwargs,          # absorbs legacy args (e.g. old sentence-transformers params)
) -> np.ndarray:
    """Generate embeddings via OpenAI API (text-embedding-3-small, 384-dim)."""
    client = _get_client()
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if show_progress:
            print(f"⚙️  Embedding batch {i // batch_size + 1}/{total_batches} ({len(batch)} texts)...")
        response = client.embeddings.create(input=batch, model=model_name, dimensions=dimensions)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    embeddings = np.array(all_embeddings, dtype=np.float32)

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms == 0, 1, norms)

    print(f"✅ Generated embeddings with shape: {embeddings.shape}")
    return embeddings


def validate_embeddings(
    embeddings: np.ndarray,
    expected_dim: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    issues = []

    if len(embeddings.shape) != 2:
        issues.append(f"❌ Wrong shape: {embeddings.shape}, expected 2D array")
        return False, issues

    if expected_dim and embeddings.shape[1] != expected_dim:
        issues.append(f"❌ Wrong dimension: {embeddings.shape[1]}, expected {expected_dim}")

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        issues.append(f"⚠️  Not L2-normalized (mean norm: {norms.mean():.6f})")

    if np.isnan(embeddings).any():
        issues.append("❌ Contains NaN values")
    if np.isinf(embeddings).any():
        issues.append("❌ Contains Inf values")

    zero_vectors = np.all(embeddings == 0, axis=1).sum()
    if zero_vectors > 0:
        issues.append(f"⚠️  {zero_vectors} all-zero embedding vectors")

    if len(embeddings) > 1:
        S = embeddings @ embeddings.T
        np.fill_diagonal(S, 0)
        if np.sum(S > 0.9999) > 0:
            issues.append(f"⚠️  {np.sum(S > 0.9999)} near-duplicate embeddings")

    return len(issues) == 0, issues
