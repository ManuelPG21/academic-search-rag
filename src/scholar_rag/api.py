"""FastAPI service: /search for ranked papers, /ask for a cited answer."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from .generation import ExtractiveLLM, OllamaLLM
from .service import ScholarSearch

DATA_DIR = Path(os.getenv("SCHOLAR_DATA", Path(__file__).resolve().parents[2] / "data"))


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    k: int = Field(10, ge=1, le=50)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(5, ge=1, le=10)


def default_llm():
    backend = os.getenv("LLM_BACKEND", "extractive")
    return OllamaLLM(os.getenv("OLLAMA_MODEL", "phi4-mini-reasoning"), os.getenv("OLLAMA_HOST", "http://localhost:11434")) \
        if backend == "ollama" else ExtractiveLLM()


def create_app(engine: ScholarSearch | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or ScholarSearch.from_disk(DATA_DIR, default_llm())
        yield

    app = FastAPI(title="arXiv RAG Search", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "documents": len(request.app.state.engine.docs)}

    @app.post("/search")
    def search(req: SearchRequest, request: Request):
        hits = request.app.state.engine.search(req.query, req.k, req.mode)
        return {"results": [{k: h[k] for k in ("id", "title", "published", "category", "score", "rank")} for h in hits]}

    @app.post("/ask")
    def ask(req: AskRequest, request: Request):
        return request.app.state.engine.ask(req.question, req.k)

    return app


app = create_app()
