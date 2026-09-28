FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY data/corpus.jsonl data/corpus.jsonl
ENV PYTHONPATH=/app/src LLM_BACKEND=extractive OLLAMA_HOST=http://host.docker.internal:11434
# Pre-compute document embeddings at build time so the container starts fast.
RUN python -c "from pathlib import Path; from scholar_rag.service import ScholarSearch; from scholar_rag.generation import ExtractiveLLM; ScholarSearch.from_disk(Path('data'), ExtractiveLLM())"
EXPOSE 8000
CMD ["uvicorn", "scholar_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
