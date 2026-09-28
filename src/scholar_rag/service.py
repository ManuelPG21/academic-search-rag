"""Search engine object shared by the API, the CLI and the evaluation harness."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .generation import LLM, build_prompt, citation_validity
from .ingest import load_corpus
from .retrieval import BM25Retriever, DenseRetriever, Encoder, HybridRetriever

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ScholarSearch:
    def __init__(self, docs: list[dict], encoder: Encoder, llm: LLM, embeddings: np.ndarray | None = None):
        self.docs = {d["id"]: d for d in docs}
        self.bm25 = BM25Retriever(docs)
        self.dense = DenseRetriever(docs, encoder, embeddings=embeddings)
        self.hybrid = HybridRetriever([self.bm25, self.dense])
        self.llm = llm

    @classmethod
    def from_disk(cls, data_dir: Path, llm: LLM, encoder: Encoder | None = None) -> "ScholarSearch":
        docs = load_corpus(data_dir / "corpus.jsonl")
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(EMBED_MODEL)
        cache = data_dir / "embeddings.npy"
        emb = np.load(cache) if cache.exists() else None
        engine = cls(docs, encoder, llm, embeddings=emb)
        if emb is None:
            np.save(cache, engine.dense.emb)
        return engine

    def retriever(self, mode: str):
        return {"bm25": self.bm25, "dense": self.dense, "hybrid": self.hybrid}[mode]

    def search(self, query: str, k: int = 10, mode: str = "hybrid") -> list[dict]:
        hits = self.retriever(mode).search(query, k)
        return [{**self.docs[h.doc_id], "score": h.score, "rank": h.rank} for h in hits]

    def ask(self, question: str, k: int = 5, mode: str = "hybrid") -> dict:
        t0 = time.perf_counter()
        docs = self.search(question, k, mode)
        t1 = time.perf_counter()
        answer = self.llm.generate(build_prompt(question, docs))
        t2 = time.perf_counter()
        return {
            "answer": answer,
            "sources": [{"n": i + 1, "id": d["id"], "title": d["title"], "url": f"https://arxiv.org/abs/{d['id']}"}
                        for i, d in enumerate(docs)],
            "citation_validity": citation_validity(answer, len(docs)),
            "latency_ms": {"retrieval": round((t1 - t0) * 1e3, 1), "generation": round((t2 - t1) * 1e3, 1)},
        }
