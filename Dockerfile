FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY aiact ./aiact
COPY data/raw ./data/raw
COPY web ./web
COPY eval ./eval

# Download the embedding model and build the index at image build time,
# so that a container starts in a couple of seconds and needs no network for retrieval.
RUN python -m aiact.retriever

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "aiact.api:app", "--host", "0.0.0.0", "--port", "8000"]
