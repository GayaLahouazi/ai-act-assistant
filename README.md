# AI Act Assistant: a grounded agent over the EU AI Act, with verified citations

![ci](https://github.com/GayaLahouazi/ai-act-assistant/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.11%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

**Problem.** Teams that build AI systems now have to know what Regulation (EU) 2024/1689 (the "AI Act")
requires of them: 113 articles, 180 recitals and 13 annexes. A general-purpose chatbot answers such
questions fluently and sometimes cites an article that says something else, or that does not exist.

**This project** is a conversational assistant that answers in French from the official text only:

- an **LLM agent with tool calling** decides when to search, when to read a full article and when to
  look up a definition, and can reformulate and search again;
- retrieval is **hybrid** (BM25 + multilingual embeddings in Qdrant, fused by Reciprocal Rank Fusion)
  over **structure-aware chunks** of the Official Journal text;
- every answer goes through a **deterministic citation check**: each reference is classified as
  *verified* (the agent read it), *mentioned* (only quoted inside a text it read) or *unsupported*
  (appears nowhere in the context, so probably invented), and the interface shows it;
- the whole thing is **measured**: a 63-question evaluation set, a retrieval ablation and an
  end-to-end benchmark, plus a [journal](docs/journal.md) of what was tried and what it changed.

```
                       ┌──────────────────────── FastAPI ────────────────────────┐
 browser  ── POST /api/chat ──►  Agent loop (max 4 tool rounds, streamed over SSE)
 (chat UI)                         │   ▲
    ▲                              ▼   │ tool results
    │                         LLM API (any OpenAI-compatible endpoint:
    │                         Mistral, Groq, OpenAI, Ollama)
    │                              │ tool calls
    │                              ▼
    │            ┌─ search_regulation ──► BM25 ─┐
    │            │                              ├─► RRF ──► top-k chunks
    │            │                      Qdrant ─┘
    │            ├─ read_source ───────► full article / annex / recital
    │            └─ lookup_definition ─► the 68 definitions of Article 3
    │                              │
    └── tokens, tool trace ◄── citation check ──► JSONL trace, /api/metrics
```

The same three tools are also exposed as an **MCP server**, so any MCP client can query the regulation.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env            # then set LLM_API_KEY (Mistral has a free tier)
python -m aiact.retriever       # parse, chunk, embed: builds the index (about 1 min, once)
uvicorn aiact.api:app           # open http://localhost:8000
```

Or with Docker (the index is built into the image):

```bash
docker compose up --build
```

Without an API key the application still runs in *extractive mode*: it shows the retrieved passages
instead of a written answer, which is enough to explore the retrieval.

| Command | What it does |
|---|---|
| `pytest` | 22 tests, offline, about 3 s (scripted LLM, hashing embedder, mocked HTTP transport) |
| `python -m eval.run_retrieval` | retrieval ablation below, no LLM needed |
| `python -m eval.run_answers --judge` | end-to-end benchmark of the agent with the configured LLM |
| `python -m aiact.mcp_server` | MCP server over stdio |

## What is implemented

| Module | Content |
|---|---|
| `aiact/ingest.py` | parser of the Official Journal XHTML: one source per article, recital and annex; chunks that never straddle two articles and carry a contextual header (article title, chapter, section); one chunk per definition of Article 3 |
| `aiact/bm25.py` | Okapi BM25 on an inverted index, tokenizer for French legal text (accents, elisions, plurals) |
| `aiact/embeddings.py` | local multilingual embeddings on CPU (ONNX Runtime through fastembed), no API key |
| `aiact/retriever.py` | Qdrant collection with payload filters, Reciprocal Rank Fusion, reference routing ("article 14" goes straight to Article 14) |
| `aiact/llm.py` | async client for OpenAI-compatible APIs written on httpx: streaming (SSE), assembly of fragmented tool calls, retries with backoff on rate limits, token accounting |
| `aiact/tools.py` | the three tools and their JSON schemas; invalid calls are returned to the model as text so it can correct itself |
| `aiact/agent.py` | the tool-calling loop, bounded, streamed as events; retrieval-only fallback when no LLM is configured |
| `aiact/guardrails.py` | citation check (verified / mentioned / unsupported), input validation |
| `aiact/api.py` | FastAPI: `/api/chat` (SSE), `/api/search`, `/api/sources/{id}`, `/api/metrics`, `/health`; conversation memory per session |
| `aiact/tracing.py` | one JSON line per request (tool calls, sources, tokens, latency, cost), aggregated metrics |
| `aiact/mcp_server.py` | the tools as a Model Context Protocol server |
| `web/` | chat interface without framework: streaming, live tool trace, colour-coded citations, side panel with the full text of a cited source |
| `eval/` | 63-question set and the two benchmarks |
| `.github/workflows/ci.yml` | lint, tests, then Docker build and a smoke test of the running container |

The agent loop and the LLM client are written by hand rather than with an orchestration framework.
The loop is sixty lines; writing it made every step explicit (what is sent, what comes back, where it
can fail) and makes it straightforward to test with a scripted model.

## Results: retrieval

`python -m eval.run_retrieval`, 58 questions with a known source (26 written in everyday words, 25 with the
vocabulary of the text, 3 definitions, 4 exact references). A hit means an expected article or annex
is among the first *k* distinct sources. Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (384 d).

| | Chunking | Ranker | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR@10 |
|---|---|---|---|---|---|---|---|
| A | fixed 1000-char windows | BM25 | 849 | 0.19 | 0.60 | 0.69 | 0.397 |
| B | fixed 1000-char windows | dense | 849 | 0.24 | 0.57 | 0.62 | 0.413 |
| C | fixed 1000-char windows | hybrid | 849 | 0.38 | 0.55 | 0.67 | 0.497 |
| D | structure-aware, 1600 chars | hybrid | 578 | 0.29 | 0.59 | 0.64 | 0.452 |
| E | D + contextual header | BM25 | 578 | 0.40 | 0.71 | 0.78 | 0.558 |
| F | D + contextual header | dense | 578 | 0.28 | 0.48 | 0.57 | 0.400 |
| G | D + contextual header | hybrid | 578 | 0.38 | 0.62 | 0.69 | 0.532 |
| H | G with 700-char chunks | dense | 950 | 0.38 | 0.57 | 0.60 | 0.479 |
| I | G with 700-char chunks | hybrid | 950 | 0.41 | 0.66 | 0.74 | 0.565 |
| J | I, binding text only (no recitals) | hybrid | 950 | 0.62 | 0.72 | 0.81 | 0.698 |
| **K** | **J + reference routing** | **hybrid** | 950 | **0.69** | **0.78** | **0.86** | **0.761** |

From the naive pipeline (C) to the one in use (K): recall@1 from 0.38 to 0.69, recall@5 from 0.67 to
0.86. What the table taught me:

- **Structure alone did not help (C → D); the header did (D → G).** A chunk cut out of Article 26 does
  not contain the words "obligations des déployeurs"; its title does.
- **The embedding model silently truncates at 128 tokens.** Two texts that differ only after that
  point get a cosine similarity of exactly 1.0. With 1600-character chunks two thirds of each chunk
  were invisible to it. Shorter chunks raised dense recall@1 from 0.28 to 0.38 (F → H).
- **Recitals crowd out articles (I → J).** They restate the articles in plainer language, so they
  match questions better, but they are not the binding text. Searching articles and annexes by
  default was the largest single gain; recitals remain available through the `scope` argument.
- **Hybrid is not automatically better than BM25** (E vs G): with a weak dense ranker, fusion can
  dilute a good lexical ranking. It wins once the dense side is fixed, and on paraphrased questions.
- **Exact references are a lookup problem**, not a similarity problem (J → K).

The eight questions still missed at 5 are all paraphrases far from the wording of the text ("mon
chatbot", "trier des CV", "sur de vrais utilisateurs"). This is where the agent matters: it can
rewrite the query with the vocabulary of the regulation and search again.

## Results: end-to-end

`python -m eval.run_answers --judge` runs the agent on the 63 questions, including 3 out-of-scope
requests (one is a prompt injection) and 2 questions the regulation cannot answer, and reports:

| Metric | Meaning |
|---|---|
| source hit | the answer cites an expected source that the agent actually read |
| grounded | at least one verified citation and no unsupported one |
| fact recall | expected figures ("35 000 000", "7 %", "15 jours") present in the answer |
| abstention | out-of-scope and unanswerable questions answered without citing anything |
| faithfulness | optional LLM-as-judge score from 1 to 5, to be read as a signal only |
| cost | tool calls, latency and tokens per question |

Results are written to `eval/results/answers-<model>.json`, one file per model, so that models and
prompt versions can be compared on the same questions.

## Design choices and limits

- **Why an agent rather than a fixed retrieve-then-generate pipeline?** Questions differ: a definition
  needs one exact lookup, "un outil de tri de CV est-il à haut risque ?" needs Article 6 *and* Annex
  III, and a greeting needs no retrieval at all. The loop is bounded (4 rounds) and the last round
  forbids tool calls, so a request always terminates.
- **Why check citations in code?** The system prompt asks the model to cite only what it read; the
  check verifies it instead of trusting it, at no extra LLM cost. It catches invented references,
  not a real article summarised wrongly: that is what the judge score estimates.
- **Conversation memory** keeps questions and answers, not tool results (too large); a follow-up
  that needs the text again triggers a new search.
- **Limits.** The corpus is the regulation as published on 12 July 2024, in French, without later
  guidelines or amendments. The evaluation set is small and written by one person. The embedding
  model is small; a larger one or a cross-encoder reranker is the obvious next experiment. This is an
  information tool, not legal advice.

## Sources

- Regulation (EU) 2024/1689, Official Journal of the European Union, [ELI](http://data.europa.eu/eli/reg/2024/1689/oj),
  © European Union, reused under Commission Decision 2011/833/EU.
- Cormack, Clarke, Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*, SIGIR 2009.
- Robertson, Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009.
- Reimers, Gurevych, *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*, EMNLP 2020.
