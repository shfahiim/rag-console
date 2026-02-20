from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional

from rag import Chunk, RAGPipeline


@dataclass
class UploadProgress:
    run_id: int = 0
    stage: str = "idle"
    failed_stage: Optional[str] = None
    message: str = "Waiting for upload."
    percent: int = 0
    active: bool = False
    error: Optional[str] = None
    updated_at: float = field(default_factory=time.time)


@dataclass
class ServerState:
    pipeline: Optional[RAGPipeline] = None
    indexed_file: Optional[str] = None
    chunk_count: int = 0
    upload_progress: UploadProgress = field(default_factory=UploadProgress)
    chunk_by_id: Dict[str, Chunk] = field(default_factory=dict)


class AppState:
    def __init__(self) -> None:
        self.server = ServerState()
        self.lock = Lock()
