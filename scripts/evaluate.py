"""Offline evaluation: compare BM25, dense and hybrid retrieval; optionally evaluate grounded generation.

Retrieval benchmark (known-item search): only abstracts are indexed; each query is a paper's *title*
and the target is that paper. Titles and abstracts share topic but rarely the exact wording, which
is what separates lexical from semantic retrieval.

Every run is logged to MLflow (sqlite:///mlflow.db) so configuration changes can be compared over time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_rag import evaluation  # noqa: E402
from scholar_rag.generation import ExtractiveLLM, OllamaLLM  # noqa: E402
from scholar_rag.service import EMBED_MODEL, ScholarSearch  # noqa: E402

REPORTS = ROOT / "reports"


def retrieval_eval(engine: ScholarSearch, n_queries: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ids = list(engine.docs)
    sample = [engine.docs[i] for i in rng.choice(ids, n_queries, replace=False)]
    results = {}
    for mode in ("bm25", "dense", "hybrid"):
        ranks, lat = [], []
        for d in sample:
            t0 = time.perf_counter()
            hits = engine.retriever(mode).search(d["title"], 10)
            lat.append((time.perf_counter() - t0) * 1e3)
            ranks.append(evaluation.rank_of(d["id"], [h.doc_id for h in hits]))
        m = evaluation.summarize(ranks)
        rr = np.array([1 / r if r and r <= 10 else 0 for r in ranks])
        m["mrr_ci_low"], m["mrr_ci_high"] = evaluation.bootstrap_ci(rr)
        m["latency_p50_ms"], m["latency_p95_ms"] = float(np.percentile(lat, 50)), float(np.percentile(lat, 95))
        results[mode] = m
        with mlflow.start_run(run_name=f"retrieval-{mode}"):
            mlflow.log_params({"mode": mode, "embed_model": EMBED_MODEL, "rrf_k0": engine.hybrid.k0,
                               "n_docs": len(engine.docs), "n_queries": n_queries, "seed": seed})
            mlflow.log_metrics(m)
    return results


def generation_eval(engine: ScholarSearch, n_questions: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + 1)
    sample = [engine.docs[i] for i in rng.choice(list(engine.docs), n_questions, replace=False)]
    validity, source_cited, gen_ms = [], [], []
    for d in sample:
        out = engine.ask(f"What problem does the work on '{d['title']}' address, and what approach does it take?", k=5)
        cited = {s["id"] for s in out["sources"] if f"[{s['n']}]" in out["answer"]}
        validity.append(out["citation_validity"] if out["citation_validity"] is not None else 0.0)
        source_cited.append(d["id"] in cited)
        gen_ms.append(out["latency_ms"]["generation"])
    res = {"citation_validity": float(np.mean(validity)), "target_paper_cited": float(np.mean(source_cited)),
           "answers_with_citations": float(np.mean([v > 0 for v in validity])),
           "generation_p50_ms": float(np.percentile(gen_ms, 50))}
    with mlflow.start_run(run_name="generation"):
        mlflow.log_params({"llm": type(engine.llm).__name__, "model": getattr(engine.llm, "model", "none"), "n_questions": n_questions})
        mlflow.log_metrics(res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--llm", choices=["none", "extractive", "ollama"], default="none")
    ap.add_argument("--questions", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("scholar-rag")
    llm = OllamaLLM() if args.llm == "ollama" else ExtractiveLLM()
    engine = ScholarSearch.from_disk(ROOT / "data", llm)

    ret = retrieval_eval(engine, args.queries, args.seed)
    report = {"corpus_size": len(engine.docs), "n_queries": args.queries, "retrieval": ret}
    if args.llm != "none":
        report["generation"] = {args.llm: generation_eval(engine, args.questions, args.seed)}

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    modes = list(ret)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(3)
    for i, mode in enumerate(modes):
        vals = [ret[mode]["recall_at_1"], ret[mode]["recall_at_10"], ret[mode]["mrr_at_10"]]
        ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=mode)
    ax.set_xticks(x, ["Recall@1", "Recall@10", "MRR@10"]); ax.set_ylim(0, 1); ax.legend()
    ax.set_title(f"Known-item retrieval, {args.queries} title queries, {len(engine.docs):,} arXiv abstracts")
    fig.tight_layout(); fig.savefig(REPORTS / "figures" / "retrieval.png", dpi=150); plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
