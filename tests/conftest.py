"""Shared fixtures. The suite runs offline: the real regulation is parsed, but embeddings come
from a deterministic hashing embedder and the LLM is scripted."""

import pytest

from aiact.config import Settings
from aiact.embeddings import HashEmbedder
from aiact.ingest import parse_regulation
from aiact.retriever import Retriever


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    tmp = tmp_path_factory.mktemp("data")
    return Settings(
        _env_file=None,
        llm_api_key="test-key",
        llm_max_retries=2,
        embedding_model="hash",
        index_dir=tmp / "index",
        trace_file=tmp / "traces.jsonl",
    )


@pytest.fixture(scope="session")
def chunks(settings):
    return parse_regulation(settings.raw_html)


@pytest.fixture(scope="session")
def retriever(chunks) -> Retriever:
    return Retriever(chunks, HashEmbedder())
