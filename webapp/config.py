from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Set


@dataclass(frozen=True)
class QuerySettings:
    dense_top_k: int = 120
    sparse_top_k: int = 120
    fuse_top_k: int = 180
    rerank_k: int = 60
    keep_n: int = 20
    token_budget: int = 3200
    retrieve_top_k: int = 20


@dataclass(frozen=True)
class AppConfig:
    max_upload_mb: int
    chunk_max_tokens: int
    chunk_overlap_tokens: int
    gemini_embed_model: str
    gemini_chat_model: str
    flask_host: str
    flask_port: int
    flask_debug: bool
    allowed_extensions: Set[str] = field(
        default_factory=lambda: {
            ".txt",
            ".md",
            ".csv",
            ".json",
            ".html",
            ".htm",
            ".xml",
            ".pdf",
            ".docx",
        }
    )
    query_settings: QuerySettings = field(default_factory=QuerySettings)

    @property
    def max_content_length(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def load_config_from_env() -> AppConfig:
    return AppConfig(
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "250")),
        chunk_max_tokens=int(os.getenv("CHUNK_MAX_TOKENS", "512")),
        chunk_overlap_tokens=int(os.getenv("CHUNK_OVERLAP_TOKENS", "32")),
        gemini_embed_model=os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004"),
        gemini_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
        flask_host=os.getenv("FLASK_HOST", "0.0.0.0"),
        flask_port=int(os.getenv("FLASK_PORT", "5000")),
        flask_debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
