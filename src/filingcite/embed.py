"""Embedding and reranking models. "hash" and "lexical" are offline stand-ins used in tests."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

from . import config

DIM_HASH = 384
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_TOK = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Bag-of-words hashing into 384 dims. Not semantic, but deterministic and instant."""

    dim = DIM_HASH
    name = "hash"

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in _TOK.findall(t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


class StEmbedder:
    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model
        self.model = SentenceTransformer(model)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        if query:
            texts = [QUERY_PREFIX + t for t in texts]
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)


@lru_cache(maxsize=2)
def get_embedder(name: str | None = None) -> HashEmbedder | StEmbedder:
    name = name or config.EMBED_MODEL
    return HashEmbedder() if name == "hash" else StEmbedder(name)


class LexicalReranker:
    """Token overlap score. Offline stand-in for the cross-encoder."""

    name = "lexical"

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(_TOK.findall(query.lower()))
        return [len(q & set(_TOK.findall(t.lower()))) / (len(q) or 1) for t in texts]


class CrossEncoderReranker:
    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder

        self.name = model
        self.model = CrossEncoder(model, max_length=384)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [
            float(s) for s in self.model.predict([(query, t) for t in texts], batch_size=16, show_progress_bar=False)
        ]


@lru_cache(maxsize=2)
def get_reranker(name: str | None = None) -> LexicalReranker | CrossEncoderReranker:
    name = name or config.RERANK_MODEL
    return LexicalReranker() if name == "lexical" else CrossEncoderReranker(name)
