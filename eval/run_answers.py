"""End-to-end benchmark of the agent on the golden set (needs an LLM: see .env.example).

    python -m eval.run_answers                 # every question
    python -m eval.run_answers --limit 10      # quick run
    python -m eval.run_answers --judge         # add an LLM-as-judge faithfulness score

Deterministic metrics (no judge involved):
  source_hit       the answer cites an expected source that the agent actually read
  grounded         at least one verified citation and no unsupported one (guardrails.py)
  fact_recall      share of the expected facts ("35 000 000", "7 %") found in the answer
  abstention       out-of-scope and unanswerable questions answered without citing any source
plus latency, tool calls and tokens per question.

Optional metric: with --judge, a second LLM call rates from 1 to 5 whether each answer is supported
by the passages the agent retrieved. A judge is itself a model: read it as a signal, not a truth.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
from pathlib import Path

from aiact.agent import Agent
from aiact.config import get_settings
from aiact.llm import ChatClient
from aiact.retriever import load_retriever

from .run_retrieval import RESULTS, load_golden

NO_ANSWER = ("out_of_scope", "unanswerable")
JUDGE_PROMPT = """Tu évalues un assistant juridique. Voici une question, les passages du règlement \
que l'assistant a consultés, et sa réponse. Note de 1 à 5 la fidélité de la réponse aux passages : \
5 = chaque affirmation est appuyée par les passages ; 3 = l'essentiel est appuyé mais un détail ne \
l'est pas ; 1 = la réponse contredit les passages ou invente. Réponds uniquement par un objet JSON : \
{{"score": <entier>, "raison": "<une phrase>"}}

QUESTION : {question}

PASSAGES :
{passages}

RÉPONSE :
{answer}"""


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ").replace(" ", " ")).lower()


async def judge(llm: ChatClient, retriever, row: dict) -> int | None:
    passages = "\n\n".join(
        f"[{retriever.source_meta(s)['label']}]\n{retriever.source_text(s)[:2500]}" for s in row["retrieved"][:6]
    )
    prompt = JUDGE_PROMPT.format(question=row["question"], passages=passages, answer=row["answer"])
    response = await llm.complete([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=120)
    match = re.search(r'"score"\s*:\s*([1-5])', response.content)
    return int(match.group(1)) if match else None


async def main(limit: int | None, with_judge: bool, pause: float) -> None:
    settings = get_settings()
    if not settings.llm_enabled:
        raise SystemExit("No LLM configured: copy .env.example to .env and set LLM_API_KEY.")
    retriever = load_retriever(settings)
    llm = ChatClient(settings)
    agent = Agent(llm, retriever, settings)
    questions = load_golden()[:limit]

    rows = []
    for q in questions:
        final = [event async for event in agent.run(q["question"])][-1]
        cited = final["citations"]["verified"]
        row = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": final["answer"],
            "expected": q["expected"],
            "cited": cited,
            "retrieved": final["retrieved"],
            "unsupported": final["citations"]["unsupported"],
            "grounded": final["citations"]["grounded"],
            "tool_calls": len(final["steps"]),
            "latency_ms": final["latency_ms"],
            "tokens": final["usage"]["prompt_tokens"] + final["usage"]["completion_tokens"],
            "error": final["error"],
        }
        if q["expected"]:
            row["source_hit"] = any(s in cited for s in q["expected"])
        if q.get("facts"):
            answer = normalise(final["answer"])
            row["fact_recall"] = sum(normalise(f) in answer for f in q["facts"]) / len(q["facts"])
        if q["category"] in NO_ANSWER:
            row["abstained"] = not cited and not final["citations"]["unsupported"]
        if with_judge and q["expected"] and final["answer"]:
            row["faithfulness"] = await judge(llm, retriever, row)
        rows.append(row)
        status = "ok " if row.get("source_hit", row.get("abstained")) else "MISS"
        print(f"{q['id']} {status} {row['latency_ms']:>6} ms  tools={row['tool_calls']}  cited={cited}")
        await asyncio.sleep(pause)  # stay under the rate limit of free tiers

    def mean(key: str, subset=None) -> float | None:
        values = [r[key] for r in (subset or rows) if r.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else None

    answerable = [r for r in rows if r["expected"]]
    summary = {
        "model": settings.llm_model,
        "questions": len(rows),
        "source_hit": mean("source_hit"),
        "source_hit_by_category": {
            c: mean("source_hit", [r for r in answerable if r["category"] == c])
            for c in sorted({r["category"] for r in answerable})
        },
        "grounded": mean("grounded", answerable),
        "answers_with_unsupported_citation": sum(bool(r["unsupported"]) for r in rows),
        "fact_recall": mean("fact_recall"),
        "abstention": mean("abstained"),
        "faithfulness_judge": mean("faithfulness"),
        "avg_tool_calls": mean("tool_calls"),
        "latency_ms_median": statistics.median(r["latency_ms"] for r in rows),
        "avg_tokens": mean("tokens"),
        "errors": sum(bool(r["error"]) for r in rows),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    RESULTS.mkdir(exist_ok=True)
    name = re.sub(r"[^a-z0-9.-]+", "-", settings.llm_model.lower())
    out: Path = RESULTS / f"answers-{name}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Written to {out}")
    await llm.aclose()
    retriever.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between questions")
    args = parser.parse_args()
    asyncio.run(main(args.limit, args.judge, args.pause))
