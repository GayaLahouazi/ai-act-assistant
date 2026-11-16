"""Observability: one JSON line per request, and aggregate metrics for /api/metrics."""

from __future__ import annotations

import json
import statistics
import time
from collections import deque
from pathlib import Path


class Tracer:
    def __init__(self, path: Path | None = None, keep: int = 500):
        self.path = path
        self.recent: deque[dict] = deque(maxlen=keep)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, trace: dict) -> None:
        trace = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **trace}
        self.recent.append(trace)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    def metrics(self) -> dict:
        traces = list(self.recent)
        if not traces:
            return {"requests": 0}
        latencies = sorted(t["latency_ms"] for t in traces)
        p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
        return {
            "requests": len(traces),
            "errors": sum(1 for t in traces if t.get("error")),
            "latency_ms_p50": statistics.median(latencies),
            "latency_ms_p95": p95,
            "grounded_rate": round(sum(t["citations"]["grounded"] for t in traces) / len(traces), 3),
            "unsupported_citations": sum(len(t["citations"]["unsupported"]) for t in traces),
            "avg_tool_calls": round(sum(len(t["steps"]) for t in traces) / len(traces), 2),
            "prompt_tokens": sum(t["usage"]["prompt_tokens"] for t in traces),
            "completion_tokens": sum(t["usage"]["completion_tokens"] for t in traces),
            "cost_eur": round(sum(t["usage"]["cost_eur"] for t in traces), 4),
        }
