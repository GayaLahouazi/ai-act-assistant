"""Central configuration, read from environment variables (or a local .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (any OpenAI-compatible chat completions endpoint) ---------------------------
    llm_base_url: str = "https://api.mistral.ai/v1"
    llm_api_key: str = ""
    llm_model: str = "mistral-small-latest"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 900
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 4
    # Optional prices in EUR per million tokens, only used to estimate the cost of a request.
    llm_price_in: float = 0.0
    llm_price_out: float = 0.0

    # --- Retrieval ------------------------------------------------------------------------
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    retrieval_mode: str = "hybrid"  # bm25 | dense | hybrid
    top_k: int = 6
    rrf_k: int = 60
    candidates: int = 30

    # --- Agent ----------------------------------------------------------------------------
    max_tool_rounds: int = 4
    history_turns: int = 4

    # --- Paths ----------------------------------------------------------------------------
    raw_html: Path = ROOT / "data" / "raw" / "ai_act_fr.html"
    index_dir: Path = ROOT / "data" / "index"
    trace_file: Path = ROOT / "data" / "traces" / "traces.jsonl"

    @property
    def llm_enabled(self) -> bool:
        """Local servers such as Ollama need no key; hosted APIs do."""
        local = any(h in self.llm_base_url for h in ("localhost", "127.0.0.1", "host.docker.internal"))
        return bool(self.llm_api_key) or local


@lru_cache
def get_settings() -> Settings:
    return Settings()
