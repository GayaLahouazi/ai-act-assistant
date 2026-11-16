import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from aiact.agent import Agent
from aiact.api import create_app
from aiact.guardrails import check_citations
from aiact.llm import ChatClient, LLMError, LLMResponse, ScriptedLLM, ToolCall
from aiact.tools import Toolbox


def collect(async_iterator) -> list[dict]:
    async def run():
        return [event async for event in async_iterator]

    return asyncio.run(run())


def call(name: str, **arguments) -> ToolCall:
    return ToolCall("call00001", name, json.dumps(arguments))


# --- tools ----------------------------------------------------------------------------------------


def test_lookup_definition_tolerates_plural_and_missing_accents(retriever):
    box = Toolbox(retriever)
    assert "«déployeur»" in box.lookup_definition("deployeurs")
    assert "n'est pas défini" in box.lookup_definition("blockchain quantique")
    assert "art-3" in box.seen


def test_read_source_reports_nonexistent_articles(retriever):
    box = Toolbox(retriever)
    assert "n'existe pas" in box.read_source("article 250")
    assert box.read_source("annexe III").startswith("[Annexe III]")
    assert list(box.seen) == ["anx-III"]


def test_invalid_tool_calls_come_back_as_text_for_the_model(retriever):
    box = Toolbox(retriever)
    assert "Appel invalide" in box.call("search_regulation", "{not json")
    assert "Appel invalide" in box.call("read_source", "{}")
    assert "Outil inconnu" in box.call("delete_everything", "{}")


# --- citation check -------------------------------------------------------------------------------


def test_citations_are_split_into_verified_mentioned_unsupported():
    answer = "Obligation de transparence [Article 50], voir aussi l'annexe III. Sanctions : [Article 99]."
    report = check_citations(answer, seen={"art-50"}, mentioned={"anx-III"})
    assert (report.verified, report.mentioned, report.unsupported) == (["art-50"], ["anx-III"], ["art-99"])
    assert not report.grounded
    assert check_citations("Voir [Article 50].", {"art-50"}, set()).grounded
    assert not check_citations("Réponse sans aucune source.", {"art-50"}, set()).grounded


# --- agent loop -----------------------------------------------------------------------------------


def test_agent_searches_then_answers_and_flags_invented_citations(retriever, settings):
    llm = ScriptedLLM([
        LLMResponse(tool_calls=[call("read_source", reference="article 50")], prompt_tokens=900, completion_tokens=20),
        LLMResponse(content="Oui, il faut informer l'utilisateur [Article 50]. Amende possible [Article 112].",
                    prompt_tokens=2500, completion_tokens=60),
    ])
    events = collect(Agent(llm, retriever, settings).run("Mon chatbot doit-il dire qu'il est une IA ?"))
    types = [e["type"] for e in events]
    assert types[:2] == ["tool_call", "tool_result"] and types[-1] == "final"

    final = events[-1]
    assert final["citations"]["verified"] == ["art-50"]
    assert final["citations"]["unsupported"] == ["art-112"]
    assert final["usage"]["prompt_tokens"] == 3400 and final["steps"][0]["sources"] == ["art-50"]
    # The second request carries the tool result back to the model.
    assert llm.calls[1]["messages"][-1]["role"] == "tool"
    assert "[Article 50]" in llm.calls[1]["messages"][-1]["content"]


def test_agent_loop_is_bounded(retriever, settings):
    endless = [LLMResponse(tool_calls=[call("search_regulation", query=f"essai {i}")])
               for i in range(settings.max_tool_rounds)]
    llm = ScriptedLLM([*endless, LLMResponse(content="Réponse finale [Article 5].")])
    final = collect(Agent(llm, retriever, settings).run("pratiques interdites"))[-1]
    assert len(final["steps"]) == settings.max_tool_rounds
    assert llm.calls[-1]["tool_choice"] == "none"  # the last round forbids further tool calls
    assert final["answer"].startswith("Réponse finale")


def test_agent_without_llm_falls_back_to_retrieved_passages(retriever, settings):
    events = collect(Agent(None, retriever, settings).run("Que dit l'article 14 ?"))
    final = events[-1]
    assert "Mode extractif" in final["answer"] and final["citations"]["verified"][0] == "art-14"


def test_history_is_passed_to_the_model(retriever, settings):
    llm = ScriptedLLM([LLMResponse(content="Bonjour.")])
    history = [{"role": "user", "content": "Bonjour"}, {"role": "assistant", "content": "Bonjour !"}]
    collect(Agent(llm, retriever, settings).run("Et les amendes ?", history))
    assert [m["role"] for m in llm.calls[0]["messages"]] == ["system", "user", "assistant", "user"]


# --- LLM client on the wire -----------------------------------------------------------------------


def sse(*chunks: dict) -> bytes:
    return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"


def test_client_assembles_fragmented_tool_calls_and_retries_on_429(settings):
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(json.loads(request.content))
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"message": "rate limited"})
        return httpx.Response(200, content=sse(
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "abc123xyz", "function": {
                "name": "read_source", "arguments": '{"refer'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ence": "article 5"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
             "usage": {"prompt_tokens": 812, "completion_tokens": 17}},
        ))

    client = ChatClient(settings, transport=httpx.MockTransport(handler))
    response = asyncio.run(client.complete([{"role": "user", "content": "test"}], tools=[{"type": "function"}]))
    assert len(attempts) == 2 and attempts[1]["stream"] is True and attempts[1]["tool_choice"] == "auto"
    assert response.tool_calls == [ToolCall("abc123xyz", "read_source", '{"reference": "article 5"}')]
    assert (response.prompt_tokens, response.completion_tokens) == (812, 17)


def test_client_streams_tokens_and_surfaces_api_errors(settings):
    ok = httpx.MockTransport(lambda r: httpx.Response(200, content=sse(
        {"choices": [{"delta": {"content": "Bon"}}]}, {"choices": [{"delta": {"content": "jour"}}]})))
    events = collect(ChatClient(settings, transport=ok).stream([{"role": "user", "content": "x"}]))
    assert [e["text"] for e in events if e["type"] == "token"] == ["Bon", "jour"]
    assert events[-1]["response"].content == "Bonjour"

    unauthorized = httpx.MockTransport(lambda r: httpx.Response(401, json={"message": "Unauthorized"}))
    with pytest.raises(LLMError, match="401"):
        asyncio.run(ChatClient(settings, transport=unauthorized).complete([{"role": "user", "content": "x"}]))


# --- HTTP API -------------------------------------------------------------------------------------


def test_api_end_to_end(retriever, settings):
    llm = ScriptedLLM([
        LLMResponse(tool_calls=[call("lookup_definition", term="déployeur")]),
        LLMResponse(content="Un déployeur utilise un système d'IA sous sa propre autorité [Article 3]."),
    ])
    with TestClient(create_app(settings, retriever, llm)) as client:
        assert client.get("/health").json()["sources"] == 306

        body = client.post("/api/chat", json={"message": "Qu'est-ce qu'un déployeur ?"}).text
        frames = [f for f in body.split("\n\n") if f]
        names = [f.split("\n")[0].removeprefix("event: ") for f in frames]
        assert names[0] == "session" and names[-1] == "final" and "token" in names
        final = json.loads(frames[-1].split("data: ", 1)[1])
        assert final["citations"]["grounded"] is True

        assert client.get("/api/sources/art-5").json()["label"] == "Article 5"
        assert client.get("/api/sources/art-999").status_code == 404
        hits = client.post("/api/search", json={"query": "marquage CE", "mode": "bm25", "k": 3}).json()["hits"]
        assert len(hits) == 3 and hits[0]["bm25_rank"] == 1
        assert client.post("/api/chat", json={"message": ""}).status_code == 422
        assert client.get("/api/metrics").json()["requests"] == 1
        assert client.get("/").status_code == 200
