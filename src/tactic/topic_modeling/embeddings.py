"""Document-embedding computation and joblib cache for aggregated docs."""

from __future__ import annotations

import os

import joblib
import numpy as np

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Cache modes:
#   "compute_if_missing": load the cache when present, otherwise compute and
#                         save it (phase 1, validation, final, apply stages).
#   "require_cached":     load the cache unconditionally (phases 2 and 3 reuse
#                         the phase-1 embeddings, and a missing cache is a
#                         hard error).
MODE_COMPUTE_IF_MISSING = "compute_if_missing"
MODE_REQUIRE_CACHED = "require_cached"


def get_device_str() -> str:
    """Return "cuda" when available, else "cpu"."""
    import torch  # imported lazily so metrics-only runs don't need torch at import time

    return str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def _check_cache_alignment(doc_embeddings: np.ndarray, docs: list, cache_path: str) -> np.ndarray:
    """A cached embeddings pkl reused against a DIFFERENT document list would
    silently mis-align every downstream fit/transform/metric, so any length
    mismatch is a hard error."""
    if len(doc_embeddings) != len(docs):
        raise ValueError(
            f"Cached document embeddings at {cache_path} hold {len(doc_embeddings)} "
            f"vectors but the current document list has {len(docs)} entries. The "
            "cache is stale (built from a different document list). Delete the "
            "pkl to recompute."
        )
    return doc_embeddings


def get_or_compute_embeddings(
    docs: list,
    cache_path: str,
    mode: str = MODE_COMPUTE_IF_MISSING,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
) -> np.ndarray:
    """Load cached document embeddings, or compute and cache them (see the
    cache modes above)."""
    if mode == MODE_REQUIRE_CACHED:
        # Unconditional load: FileNotFoundError if the cache is absent.
        return _check_cache_alignment(np.array(joblib.load(cache_path)), docs, cache_path)

    if os.path.exists(cache_path):
        print("Loading existing document embeddings...")
        return _check_cache_alignment(np.array(joblib.load(cache_path)), docs, cache_path)

    print("Computing document embeddings...")
    from sentence_transformers import SentenceTransformer

    device = get_device_str()
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
    doc_embeddings = np.array(
        embedding_model.encode(docs, show_progress_bar=True, device=device)
    )
    joblib.dump(doc_embeddings, cache_path)
    return doc_embeddings
