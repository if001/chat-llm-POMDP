# app/config/settings.py
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_", env_file=".env", extra="ignore"
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:14b"
    small_llm_model: str = "qwen2.5:3b"
    embed_model: str = "nomic-embed-text"

    # Firecrawl
    firecrawl_api_key: str = ""  # self-hostでもSDK仕様上必要な場合があるため
    firecrawl_base_url: str | None = None  # self-hostならここで上書き想定

    # Chroma
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "oneonone_memory"

    # Trace
    trace_dir: str = "./data/trace"
