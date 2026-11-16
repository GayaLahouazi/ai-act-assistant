"""Retrieval benchmark: which chunking and which ranker put the right source in the top k?

    python -m eval.run_retrieval

For every question of ``golden.jsonl`` that has an expected source, each configuration returns a
ranked list of chunks; the list is reduced to distinct sources and we measure
recall@1 / @3 / @5 (is an expected source in the first k?) and MRR@10 (mean reciprocal rank).
No LLM is involved: this isolates the retrieval stage. Results go to ``eval/results/``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from aiact.config import get_settings
from aiact.embeddings import make_embedder
from aiact.ingest import naive_chunks, parse_regulation
from aiact.retriever import Retriever

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SMALL_CHARS = 700


def load_golden() -> list[dict]:
    lines = (HERE / "golden.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def ranked_sources(retriever: Retriever, question: str, mode: str, depth: int = 10, **kwargs) -> list[str]:
    hits = retriever.search(question, k=40, mode=mode, **kwargs)
    sources: dict[str, None] = {}
    for hit in hits:
        sources.setdefault(hit.chunk.source_id)
    return list(sources)[:depth]


def evaluate(retriever: Retriever, questions: list[dict], mode: str, **kwargs) -> dict:
    ranks, per_question, started = [], {}, time.perf_counter()
    for q in questions:
        sources = ranked_sources(retriever, q["question"], mode, **kwargs)
        rank = next((i for i, s in enumerate(sources, 1) if s in q["expected"]), None)
        ranks.append(rank)
        per_question[q["id"]] = rank
    n = len(questions)
    return {
        "recall@1": sum(r is not None and r <= 1 for r in ranks) / n,
        "recall@3": sum(r is not None and r <= 3 for r in ranks) / n,
        "recall@5": sum(r is not None and r <= 5 for r in ranks) / n,
        "mrr@10": sum(1 / r for r in ranks if r) / n,
        "ms_per_query": (time.perf_counter() - started) * 1000 / n,
        "ranks": per_question,
    }


def main() -> None:
    settings = get_settings()
    questions = [q for q in load_golden() if q["expected"]]
    embedder = make_embedder(settings.embedding_model)
    structured = parse_regulation(settings.raw_html, max_chars=1600)
    naive = naive_chunks(structured)
    small = parse_regulation(settings.raw_html, max_chars=SMALL_CHARS)
    binding = {"kinds": ["article", "definition", "annex"]}
    off = {"route_references": False}

    # Read top to bottom: each block changes one thing.
    configs = [
        ("A", "fixed 1000-char windows", "bm25", naive, False, off),
        ("B", "fixed 1000-char windows", "dense", naive, False, off),
        ("C", "fixed 1000-char windows", "hybrid", naive, False, off),
        ("D", "structure-aware, 1600 chars", "hybrid", structured, False, off),
        ("E", "D + contextual header", "bm25", structured, True, off),
        ("F", "D + contextual header", "dense", structured, True, off),
        ("G", "D + contextual header", "hybrid", structured, True, off),
        ("H", f"G with {SMALL_CHARS}-char chunks", "dense", small, True, off),
        ("I", f"G with {SMALL_CHARS}-char chunks", "hybrid", small, True, off),
        ("J", "I, binding text only (no recitals)", "hybrid", small, True, {**off, **binding}),
        ("K", "J + reference routing", "hybrid", small, True, binding),
    ]

    rows, cache = [], {}
    for key, chunking, mode, chunks, header, kwargs in configs:
        cache_key = (id(chunks), header)
        if cache_key not in cache:  # embed each corpus variant once
            cache[cache_key] = Retriever(chunks, embedder, with_header=header,
                                         rrf_k=settings.rrf_k, candidates=settings.candidates)
        metrics = evaluate(cache[cache_key], questions, mode, **kwargs)
        rows.append({"config": key, "chunking": chunking, "ranker": mode, "chunks": len(chunks), **metrics})
        print(f"{key}  {chunking:38s} {mode:7s} R@1={metrics['recall@1']:.2f} R@3={metrics['recall@3']:.2f} "
              f"R@5={metrics['recall@5']:.2f} MRR={metrics['mrr@10']:.3f}  {metrics['ms_per_query']:.0f} ms")

    RESULTS.mkdir(exist_ok=True)
    report = {"questions": len(questions), "embedding_model": settings.embedding_model, "rows": rows}
    (RESULTS / "retrieval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    table = [
        f"Retrieval benchmark on {len(questions)} questions, embeddings: `{settings.embedding_model}`.",
        "",
        "| | Chunking | Ranker | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR@10 | ms/query |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        table.append(
            f"| {r['config']} | {r['chunking']} | {r['ranker']} | {r['chunks']} | {r['recall@1']:.2f} | "
            f"{r['recall@3']:.2f} | {r['recall@5']:.2f} | {r['mrr@10']:.3f} | {r['ms_per_query']:.0f} |"
        )
    (RESULTS / "retrieval.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    best = rows[-1]["ranks"]
    missed = [q["id"] for q in questions if not best[q["id"]] or best[q["id"]] > 5]
    print("Missed at 5 by the last configuration:", missed)


if __name__ == "__main__":
    main()
