"""Minimal async client for OpenAI-compatible chat completion APIs (Mistral, Groq, OpenAI, Ollama...).

Written directly on top of httpx rather than through an SDK or an orchestration framework, to keep
every step of the protocol visible: the JSON payload, server-sent events, the incremental assembly
of tool calls, retries on rate limits and token accounting.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from .config import Settings

RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, exactly as produced by the model

    def as_message_part(self) -> dict:
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""


class LLMError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.s = settings
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
        self.http = httpx.AsyncClient(
            base_url=settings.llm_base_url.rstrip("/") + "/",
            headers=headers,
            timeout=settings.llm_timeout_s,
            transport=transport,
        )

    @property
    def model(self) -> str:
        return self.s.llm_model

    def _payload(self, messages: list[dict], tools: list[dict] | None, **overrides) -> dict:
        payload = {
            "model": self.s.llm_model,
            "messages": messages,
            "temperature": self.s.llm_temperature,
            "max_tokens": self.s.llm_max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if "mistral" not in self.s.llm_base_url:
            # OpenAI-style servers only report token usage in a stream when asked to;
            # Mistral always sends it in the last chunk and rejects unknown fields.
            payload["stream_options"] = {"include_usage": True}
        payload.update(overrides)
        return payload

    async def stream(self, messages: list[dict], tools: list[dict] | None = None, **overrides) -> AsyncIterator[dict]:
        """Yield ``{"type": "token", "text": ...}`` while the model writes, then a final
        ``{"type": "response", "response": LLMResponse}``."""
        payload = self._payload(messages, tools, **overrides)
        for attempt in range(self.s.llm_max_retries + 1):
            try:
                async with self.http.stream("POST", "chat/completions", json=payload) as r:
                    if r.status_code in RETRYABLE and attempt < self.s.llm_max_retries:
                        await asyncio.sleep(self._backoff(attempt, r.headers.get("retry-after")))
                        continue
                    if r.status_code >= 400:
                        body = (await r.aread()).decode(errors="replace")[:500]
                        raise LLMError(f"LLM API returned HTTP {r.status_code}: {body}")
                    async for event in self._parse_sse(r):
                        yield event
                    return
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                if attempt >= self.s.llm_max_retries:
                    raise LLMError(f"LLM API unreachable: {exc!r}") from exc
                await asyncio.sleep(self._backoff(attempt, None))

    async def complete(self, messages: list[dict], tools: list[dict] | None = None, **overrides) -> LLMResponse:
        """Non-streaming convenience wrapper (used by the LLM judge of the evaluation)."""
        response = LLMResponse()
        async for event in self.stream(messages, tools, **overrides):
            if event["type"] == "response":
                response = event["response"]
        return response

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        """Exponential backoff with jitter; the server's Retry-After header wins when present."""
        if retry_after and retry_after.replace(".", "", 1).isdigit():
            return min(float(retry_after), 30.0)
        return min(2.0**attempt, 16.0) + random.uniform(0, 0.5)

    @staticmethod
    async def _parse_sse(r: httpx.Response) -> AsyncIterator[dict]:
        response = LLMResponse()
        partial: dict[int, dict] = {}  # tool calls arrive in fragments, keyed by their index
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage")
            if usage:
                response.prompt_tokens = usage.get("prompt_tokens", 0)
                response.completion_tokens = usage.get("completion_tokens", 0)
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if isinstance(text, list):  # some models send a list of typed parts
                    text = "".join(p.get("text", "") for p in text if p.get("type") == "text")
                if text:
                    response.content += text
                    yield {"type": "token", "text": text}
                for position, call in enumerate(delta.get("tool_calls") or []):
                    slot = partial.setdefault(call.get("index", position), {"id": "", "name": "", "arguments": ""})
                    fn = call.get("function") or {}
                    slot["id"] = call.get("id") or slot["id"]
                    slot["name"] = fn.get("name") or slot["name"]
                    arguments = fn.get("arguments") or ""
                    slot["arguments"] += arguments if isinstance(arguments, str) else json.dumps(arguments)
                if choice.get("finish_reason"):
                    response.finish_reason = choice["finish_reason"]
        response.tool_calls = [
            ToolCall(slot["id"] or f"call{index:05d}", slot["name"], slot["arguments"] or "{}")
            for index, slot in sorted(partial.items())
            if slot["name"]
        ]
        yield {"type": "response", "response": response}

    async def aclose(self) -> None:
        await self.http.aclose()


class ScriptedLLM:
    """Stand-in for a real model: replays a fixed list of responses. Used by the tests to exercise
    the agent loop deterministically, without network and without an API key."""

    model = "scripted"

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def stream(self, messages: list[dict], tools: list[dict] | None = None, **overrides) -> AsyncIterator[dict]:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools, **overrides})
        response = self.responses.pop(0)
        for word in response.content.split(" ") if response.content else []:
            yield {"type": "token", "text": word + " "}
        yield {"type": "response", "response": response}

    async def complete(self, messages: list[dict], tools: list[dict] | None = None, **_) -> LLMResponse:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.responses.pop(0)
