"""Download paper metadata from the public arXiv API into a JSONL corpus."""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}
DEFAULT_CATEGORIES = ["physics.soc-ph", "nlin.AO", "cs.IR", "stat.ML"]


def fetch_category(category: str, n: int = 800, page: int = 200, pause: float = 3.0) -> list[dict]:
    """Most recent `n` papers of an arXiv category (the API asks for ~3 s between calls)."""
    papers = []
    for start in range(0, n, page):
        params = {"search_query": f"cat:{category}", "start": start, "max_results": page,
                  "sortBy": "submittedDate", "sortOrder": "descending"}
        for attempt in range(5):
            resp = requests.get(API, params=params, timeout=60)
            if resp.ok:
                break
            time.sleep(pause * (attempt + 2))
        resp.raise_for_status()
        entries = ET.fromstring(resp.content).findall("a:entry", NS)
        for e in entries:
            papers.append({
                "id": e.findtext("a:id", namespaces=NS).rsplit("/", 1)[-1],
                "title": " ".join(e.findtext("a:title", namespaces=NS).split()),
                "abstract": " ".join(e.findtext("a:summary", namespaces=NS).split()),
                "authors": [a.findtext("a:name", namespaces=NS) for a in e.findall("a:author", NS)],
                "published": e.findtext("a:published", namespaces=NS)[:10],
                "category": category,
            })
        if len(entries) < page:
            break
        time.sleep(pause)
    return papers


def build_corpus(path: Path, categories: list[str] = DEFAULT_CATEGORIES, per_category: int = 800) -> int:
    """Write a de-duplicated JSONL corpus and return the number of papers."""
    seen, rows = set(), []
    for cat in categories:
        for p in fetch_category(cat, per_category):
            base_id = p["id"].split("v")[0]
            if base_id not in seen and len(p["abstract"]) > 200:
                seen.add(base_id)
                rows.append(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def load_corpus(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]
