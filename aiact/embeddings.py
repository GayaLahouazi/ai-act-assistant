"""Text embedders. All return L2-normalised vectors, so a dot product is a cosine similarity."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .bm25 import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class FastEmbedEmbedder:
    """Multilingual sentence embeddings running locally on CPU with ONNX Runtime (no API key)."""

    def __init__(self, model: str):
        from fastembed import TextEmbedding  # imported lazily: loading ONNX Runtime takes a second

        self.name = model
        self._model = TextEmbedding(model_name=model)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self._model.embed(texts, batch_size=32)), dtype=np.float32)
        return _normalise(vectors)


class HashEmbedder:
    """Deterministic bag-of-words hashing. Not semantic: used by the test suite, which must run
    offline and in a few seconds."""

    def __init__(self, dim: int = 256):
        self.name, self.dim = f"hash-{dim}", dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                bucket = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little") % self.dim
                vectors[row, bucket] += 1.0
        return _normalise(vectors)


def make_embedder(model: str) -> Embedder:
    return HashEmbedder() if model.startswith("hash") else FastEmbedEmbedder(model)
