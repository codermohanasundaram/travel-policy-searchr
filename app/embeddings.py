"""
embeddings.py
-------------
Responsible for ONE job: turning text into a vector (list of numbers).

Primary path: sentence-transformers (all-MiniLM-L6-v2)
    - A real, production-grade embedding model.
    - Downloads (~90MB) the first time you run it, so you need
      internet access on your machine once.

Fallback path: a tiny hashing-based embedder
    - Used automatically ONLY if sentence-transformers / internet is
      unavailable.
    - NOT real semantic understanding — a bag-of-character-ngrams
      hash. It exists purely so the pipeline still runs end-to-end
      even without the real model.
"""

import hashlib
import numpy as np

_MODEL = None
_USING_REAL_MODEL = False
_DIM = 384  # matches all-MiniLM-L6-v2 output size


def _load_real_model():
    global _MODEL, _USING_REAL_MODEL
    if _MODEL is not None:
        return _MODEL
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    _USING_REAL_MODEL = True
    return _MODEL


def _fallback_embed(text: str, dim: int = _DIM) -> np.ndarray:
    text = text.lower().strip()
    vec = np.zeros(dim, dtype=np.float32)
    ngrams = [text[i:i + 3] for i in range(max(len(text) - 2, 1))]
    if not ngrams:
        ngrams = [text]
    for ng in ngrams:
        h = int(hashlib.md5(ng.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def get_embedding(text: str) -> np.ndarray:
    """One string in, one vector out. Use for a single query."""
    try:
        model = _load_real_model()
        return np.array(model.encode(text), dtype=np.float32)
    except Exception as e:
        print(f"Real model failed to load, using fallback: {e}")
        return _fallback_embed(text)


def get_embeddings(texts: list[str]) -> np.ndarray:
    """A list of strings in, a 2D array of vectors out. Use for a batch of documents."""
    try:
        model = _load_real_model()
        return np.array(model.encode(texts), dtype=np.float32)
    except Exception as e:
        print(f"Real model failed to load, using fallback: {e}")
        return np.vstack([_fallback_embed(t) for t in texts])


def using_real_model() -> bool:
    """Lets callers (and /health) know which backend is active."""
    return _USING_REAL_MODEL