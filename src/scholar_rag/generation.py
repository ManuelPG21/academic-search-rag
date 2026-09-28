"""Grounded answer generation with citations, via a local LLM (Ollama) or an extractive fallback."""

from __future__ import annotations

import re
from typing import Protocol

import requests

PROMPT = """You are a research assistant. Answer the question using ONLY the numbered paper abstracts below.
Cite every claim with the paper number in square brackets, e.g. [2]. If the abstracts do not contain the
answer, say so. Keep the answer under 120 words.

{context}

Question: {question}
Answer:"""


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


class OllamaLLM:
    def __init__(self, model: str = "phi4-mini-reasoning", host: str = "http://localhost:11434", timeout: int = 300):
        self.model, self.host, self.timeout = model, host, timeout

    def generate(self, prompt: str) -> str:
        r = requests.post(f"{self.host}/api/generate", json={"model": self.model, "prompt": prompt, "stream": False,
                                                            "options": {"temperature": 0}}, timeout=self.timeout)
        r.raise_for_status()
        text = r.json()["response"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


class ExtractiveLLM:
    """No-model fallback: returns the first sentence of each retrieved abstract with its citation."""

    def generate(self, prompt: str) -> str:
        blocks = re.findall(r"\[(\d+)\] [^\n]*\n([^\n]+)", prompt)
        return " ".join(f"{re.split(r'(?<=[.!?]) ', abstract)[0]} [{n}]" for n, abstract in blocks[:3])


def build_prompt(question: str, docs: list[dict]) -> str:
    context = "\n\n".join(f"[{i + 1}] {d['title']}\n{d['abstract']}" for i, d in enumerate(docs))
    return PROMPT.format(context=context, question=question)


def cited_indices(answer: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer)}


def citation_validity(answer: str, n_docs: int) -> float | None:
    """Share of citations that point to a document actually shown to the model (hallucinated refs score < 1)."""
    cites = cited_indices(answer)
    if not cites:
        return None
    return sum(1 <= c <= n_docs for c in cites) / len(cites)
