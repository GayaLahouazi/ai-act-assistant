"""Hybrid retrieval: BM25 (lexical) + dense vectors in Qdrant, fused with Reciprocal Rank Fusion.

Why hybrid? Legal questions mix two needs. "Que dit l'article 50 ?" or "amende de 35 000 000 EUR"
need exact term matching, which dense embeddings handle badly. "Mon chatbot doit-il dire qu'il
est une IA ?" shares almost no word with the text of Article 50 and needs semantic matching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, PointStruct, VectorParams

from .bm25 import BM25
from .config import Settings
from .embeddings import Embedder, make_embedder
from .ingest import MAX_CHARS, Chunk, find_references, parse_regulation

COLLECTION = "ai_act"


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        client: QdrantClient | None = None,
        with_header: bool = True,
        rrf_k: int = 60,
        candidates: int = 30,
    ):
        self.chunks, self.embedder = chunks, embedder
        self.rrf_k, self.candidates = rrf_k, candidates
        self.kinds = np.array([c.kind for c in chunks])
        texts = [c.indexed_text(with_header) for c in chunks]
        self.bm25 = BM25(texts)
        self.client = client or QdrantClient(":memory:")
        if not self.client.collection_exists(COLLECTION):
            self._index(texts)

        self.by_source: dict[str, list[Chunk]] = {}
        self._first_chunk: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            self.by_source.setdefault(c.source_id, []).append(c)
            self._first_chunk.setdefault(c.source_id, [i])

    def _index(self, texts: list[str]) -> None:
        self.client.create_collection(
            COLLECTION, vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE)
        )
        vectors = self.embedder.embed(texts)
        points = [
            PointStruct(id=i, vector=vectors[i].tolist(), payload={"chunk_id": c.id, "kind": c.kind})
            for i, c in enumerate(self.chunks)
        ]
        for start in range(0, len(points), 256):
            self.client.upsert(COLLECTION, points=points[start : start + 256])

    # --- the two rankers ----------------------------------------------------------------------

    def _bm25_ranking(self, query: str, kinds: list[str] | None, n: int) -> list[int]:
        scores = self.bm25.scores(query)
        if kinds:
            scores = np.where(np.isin(self.kinds, kinds), scores, 0.0)
        order = np.argsort(-scores)[:n]
        return [int(i) for i in order if scores[i] > 0]

    def _dense_ranking(self, query: str, kinds: list[str] | None, n: int) -> list[int]:
        vector = self.embedder.embed([query])[0].tolist()
        flt = Filter(must=[FieldCondition(key="kind", match=MatchAny(any=kinds))]) if kinds else None
        result = self.client.query_points(COLLECTION, query=vector, limit=n, query_filter=flt)
        return [int(p.id) for p in result.points]

    # --- public API ----------------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 6,
        mode: str = "hybrid",
        kinds: list[str] | None = None,
        route_references: bool = True,
    ) -> list[Hit]:
        """Return the k best chunks. ``mode`` is "bm25", "dense" or "hybrid"."""
        n = max(self.candidates, k)
        lexical = self._bm25_ranking(query, kinds, n) if mode in ("bm25", "hybrid") else []
        dense = self._dense_ranking(query, kinds, n) if mode in ("dense", "hybrid") else []
        bm25_rank = {doc: r for r, doc in enumerate(lexical, 1)}
        dense_rank = {doc: r for r, doc in enumerate(dense, 1)}

        # Reciprocal Rank Fusion: score(d) = sum over rankers of 1 / (k + rank(d)).
        # It only uses ranks, so BM25 scores and cosine similarities need no calibration.
        fused: dict[int, float] = {}
        for ranking in (bm25_rank, dense_rank):
            for doc, rank in ranking.items():
                fused[doc] = fused.get(doc, 0.0) + 1.0 / (self.rrf_k + rank)
        best = sorted(fused, key=lambda d: (-fused[d], d))

        # Reference routing: "que dit l'article 14 ?" is a lookup, not a similarity problem.
        # Sources named explicitly in the query go first, whatever the rankers think.
        routed = []
        if route_references:
            for source_id in find_references(query):
                routed += [d for d in self._first_chunk.get(source_id, []) if d not in routed]
        best = (routed + [d for d in best if d not in routed])[:k]
        return [Hit(self.chunks[d], fused.get(d, 1.0), bm25_rank.get(d), dense_rank.get(d)) for d in best]

    def close(self) -> None:
        self.client.close()

    def source_text(self, source_id: str) -> str | None:
        group = self.by_source.get(source_id)
        if not group:
            return None
        if group[0].kind == "definition":  # Article 3: list the defined terms rather than 68 definitions
            return "\n".join(c.text.split(": ", 1)[-1] for c in group)
        return "\n".join(c.text for c in group)

    def source_meta(self, source_id: str) -> dict | None:
        group = self.by_source.get(source_id)
        if not group:
            return None
        first = group[0]
        return {"source_id": source_id, "label": first.label, "title": first.title, "context": first.context}


# ------------------------------------------------------------------------------------------------
# Building / loading the on-disk index
# ------------------------------------------------------------------------------------------------


def load_retriever(settings: Settings, rebuild: bool = False) -> Retriever:
    """Load the persisted index, or build it (parse, chunk, embed) on first start."""
    index_dir: Path = settings.index_dir
    meta_file, chunks_file = index_dir / "meta.json", index_dir / "chunks.jsonl"
    meta = {"embedding_model": settings.embedding_model, "source": settings.raw_html.name, "max_chars": MAX_CHARS}

    fresh = meta_file.exists() and chunks_file.exists() and json.loads(meta_file.read_text()) == meta
    if rebuild or not fresh:
        import shutil

        shutil.rmtree(index_dir, ignore_errors=True)
        index_dir.mkdir(parents=True)
        chunks = parse_regulation(settings.raw_html)
        with chunks_file.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    else:
        chunks = [Chunk(**json.loads(line)) for line in chunks_file.read_text(encoding="utf-8").splitlines()]

    retriever = Retriever(
        chunks,
        make_embedder(settings.embedding_model),
        QdrantClient(path=str(index_dir / "qdrant")),
        rrf_k=settings.rrf_k,
        candidates=settings.candidates,
    )
    meta_file.write_text(json.dumps(meta))
    return retriever


if __name__ == "__main__":
    from .config import get_settings

    r = load_retriever(get_settings(), rebuild=True)
    print(f"Indexed {len(r.chunks)} chunks from {len(r.by_source)} sources with {r.embedder.name}.")
    r.close()
