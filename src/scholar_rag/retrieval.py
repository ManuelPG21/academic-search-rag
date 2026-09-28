"""Lexical (BM25), dense (sentence embeddings) and hybrid (reciprocal rank fusion) retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were which with we our".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class Encoder(Protocol):
    def encode(self, texts: list[str], batch_size: int = ..., normalize_embeddings: bool = ...,
               show_progress_bar: bool = ...) -> np.ndarray: ...


@dataclass
class Hit:
    doc_id: str
    score: float
    rank: int


class BM25Retriever:
    def __init__(self, docs: list[dict], field: str = "abstract"):
        self.ids = [d["id"] for d in docs]
        self.bm25 = BM25Okapi([tokenize(d[field]) for d in docs])

    def search(self, query: str, k: int = 10) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [Hit(self.ids[i], float(scores[i]), r + 1) for r, i in enumerate(top)]


class DenseRetriever:
    """Cosine similarity over normalized sentence embeddings (brute force; fine up to ~1e5 docs)."""

    def __init__(self, docs: list[dict], encoder: Encoder, field: str = "abstract", embeddings: np.ndarray | None = None):
        self.ids = [d["id"] for d in docs]
        self.encoder = encoder
        self.emb = embeddings if embeddings is not None else encoder.encode(
            [d[field] for d in docs], batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        q = self.encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.emb @ q
        top = np.argsort(-scores)[:k]
        return [Hit(self.ids[i], float(scores[i]), r + 1) for r, i in enumerate(top)]


class HybridRetriever:
    """Reciprocal rank fusion: score(d) = sum_r 1 / (k0 + rank_r(d)). Robust to incomparable score scales."""

    def __init__(self, retrievers: list, k0: int = 60, depth: int = 50):
        self.retrievers, self.k0, self.depth = retrievers, k0, depth

    def search(self, query: str, k: int = 10) -> list[Hit]:
        fused: dict[str, float] = {}
        for r in self.retrievers:
            for h in r.search(query, self.depth):
                fused[h.doc_id] = fused.get(h.doc_id, 0.0) + 1.0 / (self.k0 + h.rank)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(doc_id, score, r + 1) for r, (doc_id, score) in enumerate(top)]
