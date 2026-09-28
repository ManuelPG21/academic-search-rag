import sys
import zlib
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholar_rag import evaluation  # noqa: E402
from scholar_rag.api import create_app  # noqa: E402
from scholar_rag.generation import ExtractiveLLM, citation_validity  # noqa: E402
from scholar_rag.retrieval import BM25Retriever, HybridRetriever, tokenize  # noqa: E402
from scholar_rag.service import ScholarSearch  # noqa: E402

DOCS = [
    {"id": "1", "title": "Synchronization on networks", "abstract": "Kuramoto oscillators coupled on complex networks reach synchronization.", "published": "2025-01-01", "category": "nlin.AO"},
    {"id": "2", "title": "Dense retrieval", "abstract": "Neural embeddings improve document retrieval for search engines.", "published": "2025-01-02", "category": "cs.IR"},
    {"id": "3", "title": "Epidemics", "abstract": "Epidemic spreading on contact networks with heterogeneous degree.", "published": "2025-01-03", "category": "physics.soc-ph"},
]


class HashEncoder:
    """Deterministic bag-of-words encoder so tests run without downloading a model."""

    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False):
        out = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, zlib.crc32(tok.encode()) % 64] += 1  # stable across processes, unlike hash()
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def engine():
    return ScholarSearch(DOCS, HashEncoder(), ExtractiveLLM())


def test_bm25_finds_lexical_match():
    assert BM25Retriever(DOCS).search("kuramoto oscillators", 1)[0].doc_id == "1"


def test_hybrid_fuses_rankings():
    e = engine()
    hits = HybridRetriever([e.bm25, e.dense]).search("epidemic spreading networks", 3)
    assert hits[0].doc_id == "3"
    assert [h.rank for h in hits] == [1, 2, 3]


def test_metrics_known_values():
    m = evaluation.summarize([1, 2, None, 10])
    assert m["recall_at_1"] == 0.25 and m["recall_at_10"] == 0.75
    assert np.isclose(m["mrr_at_10"], (1 + 0.5 + 0 + 0.1) / 4)


def test_citation_validity_flags_hallucinated_reference():
    assert citation_validity("A [1] and B [7].", n_docs=5) == 0.5
    assert citation_validity("no citations", n_docs=5) is None


def test_api_search_and_ask():
    with TestClient(create_app(engine())) as client:
        assert client.get("/health").json()["documents"] == 3
        r = client.post("/search", json={"query": "neural embeddings retrieval", "k": 2, "mode": "bm25"})
        assert r.status_code == 200 and r.json()["results"][0]["id"] == "2"
        a = client.post("/ask", json={"question": "How do oscillators synchronize?", "k": 2}).json()
        assert a["sources"] and a["citation_validity"] == 1.0
        assert client.post("/search", json={"query": "x"}).status_code == 422
