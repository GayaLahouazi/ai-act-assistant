"""HTTP API (FastAPI) and static web interface.

    POST /api/chat            question -> server-sent events (tool calls, tokens, final report)
    POST /api/search          retrieval playground: compare bm25 / dense / hybrid
    GET  /api/sources/{id}    full text of an article, recital or annex
    GET  /api/metrics         latency, tokens, grounded rate over recent requests
    GET  /health
"""

from __future__ import annotations

import json
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .agent import Agent
from .config import Settings, get_settings
from .guardrails import MAX_QUESTION_CHARS, validate_question
from .llm import ChatClient
from .retriever import Retriever, load_retriever
from .tracing import Tracer

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_SESSIONS = 200


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    k: int = Field(default=6, ge=1, le=20)


class Sessions:
    """In-memory conversation store. Only questions and final answers are kept: tool results are
    large, and the agent can search again if a follow-up needs them."""

    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        self._store: OrderedDict[str, list[dict]] = OrderedDict()

    def history(self, session_id: str) -> list[dict]:
        return list(self._store.get(session_id, []))

    def append(self, session_id: str, question: str, answer: str) -> None:
        turns = self._store.setdefault(session_id, [])
        turns += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        del turns[: -2 * self.max_turns]
        self._store.move_to_end(session_id)
        while len(self._store) > MAX_SESSIONS:
            self._store.popitem(last=False)


def create_app(
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    llm=None,
) -> FastAPI:
    """Application factory. Tests inject a small retriever and a scripted LLM."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.retriever = retriever or load_retriever(settings)
        client = llm or (ChatClient(settings) if settings.llm_enabled else None)
        app.state.tracer = Tracer(settings.trace_file)
        app.state.agent = Agent(client, app.state.retriever, settings, app.state.tracer)
        app.state.sessions = Sessions(settings.history_turns)
        yield
        if isinstance(client, ChatClient):
            await client.aclose()
        if retriever is None:
            app.state.retriever.close()

    app = FastAPI(title="AI Act Assistant", version=__version__, lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request) -> dict:
        r: Retriever = request.app.state.retriever
        return {
            "status": "ok",
            "version": __version__,
            "chunks": len(r.chunks),
            "sources": len(r.by_source),
            "embedding_model": r.embedder.name,
            "llm_model": settings.llm_model if request.app.state.agent.llm else None,
            "retrieval_mode": settings.retrieval_mode,
        }

    @app.post("/api/chat")
    async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
        try:
            question = validate_question(body.message)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        session_id = body.session_id or uuid.uuid4().hex
        sessions: Sessions = request.app.state.sessions
        agent: Agent = request.app.state.agent

        async def events():
            yield _sse("session", {"session_id": session_id})
            async for event in agent.run(question, sessions.history(session_id)):
                if event["type"] == "final" and event["answer"] and not event["error"]:
                    sessions.append(session_id, question, event["answer"])
                yield _sse(event["type"], {k: v for k, v in event.items() if k != "type"})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/search")
    async def search(body: SearchRequest, request: Request) -> dict:
        r: Retriever = request.app.state.retriever
        hits = r.search(body.query, body.k, body.mode)
        return {
            "mode": body.mode,
            "hits": [
                {
                    "chunk_id": h.chunk.id,
                    "source_id": h.chunk.source_id,
                    "label": h.chunk.label,
                    "title": h.chunk.title,
                    "text": h.chunk.text,
                    "score": round(h.score, 5),
                    "bm25_rank": h.bm25_rank,
                    "dense_rank": h.dense_rank,
                }
                for h in hits
            ],
        }

    @app.get("/api/sources/{source_id}")
    async def source(source_id: str, request: Request) -> dict:
        r: Retriever = request.app.state.retriever
        meta = r.source_meta(source_id)
        if meta is None:
            raise HTTPException(404, f"Unknown source '{source_id}'")
        return {**meta, "text": r.source_text(source_id)}

    @app.get("/api/metrics")
    async def metrics(request: Request) -> dict:
        return request.app.state.tracer.metrics()

    if WEB_DIR.exists():

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    return app


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


app = create_app()
