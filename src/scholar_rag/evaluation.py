"""Ranking metrics for known-item retrieval (one relevant document per query)."""

from __future__ import annotations

import numpy as np


def rank_of(relevant_id: str, ranked_ids: list[str]) -> int | None:
    try:
        return ranked_ids.index(relevant_id) + 1
    except ValueError:
        return None


def summarize(ranks: list[int | None], ks=(1, 5, 10)) -> dict:
    """Recall@k, MRR@10 and nDCG@10 for single-relevant-document queries."""
    r = np.array([x if x is not None else np.inf for x in ranks], dtype=float)
    out = {f"recall_at_{k}": float(np.mean(r <= k)) for k in ks}
    out["mrr_at_10"] = float(np.mean(np.where(r <= 10, 1 / r, 0)))
    out["ndcg_at_10"] = float(np.mean(np.where(r <= 10, 1 / np.log2(r + 1), 0)))
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values)).mean() for _ in range(n_boot)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
