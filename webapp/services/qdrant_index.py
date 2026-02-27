from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import numpy as np

from rag import Chunk


@dataclass
class QdrantIndexConfig:
    url: str
    collection: str = "rag_chunks"
    api_key: Optional[str] = None
    recreate_collection_on_first_add: bool = True
    timeout_s: float = 20.0


class QdrantCosineIndex:
    """
    Minimal Qdrant-backed dense index implementing the `VectorIndex` protocol from `rag.py`.

    Uses Qdrant's HTTP API via stdlib to avoid extra dependencies.

    Important: This app's pipeline assumes "internal_index" returned by search maps to
    `pipeline._chunks[internal_index]`. To preserve that contract, this index uses
    sequential integer point IDs starting at 0, matching append order.
    """

    def __init__(self, config: QdrantIndexConfig) -> None:
        self._config = config
        self._count = 0
        self._initialized = False
        self._vector_size: Optional[int] = None

        base = (self._config.url or "").strip()
        if not base:
            raise ValueError("Qdrant url is required.")
        if not base.endswith("/"):
            base += "/"
        self._base_url = base

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["api-key"] = self._config.api_key
        return headers

    def _request_json(self, method: str, path: str, body: Optional[bytes] = None) -> Dict[str, Any]:
        import json

        url = urljoin(self._base_url, path.lstrip("/"))
        req = Request(url, data=body, method=method, headers=self._headers())
        try:
            with urlopen(req, timeout=self._config.timeout_s) as resp:  # nosec - caller controls url
                raw = resp.read() or b"{}"
        except Exception as exc:
            raise RuntimeError(f"Qdrant HTTP error calling {method} {url}: {exc}") from exc

        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def _ensure_collection(self, vector_size: int) -> None:
        import json

        if self._initialized:
            if self._vector_size is not None and self._vector_size != vector_size:
                raise RuntimeError(
                    f"Qdrant collection vector size mismatch: existing={self._vector_size} new={vector_size}."
                )
            return

        if self._config.recreate_collection_on_first_add:
            try:
                self._request_json("DELETE", f"/collections/{self._config.collection}")
            except Exception:
                pass

        # Check existence
        exists = True
        try:
            self._request_json("GET", f"/collections/{self._config.collection}")
        except Exception:
            exists = False

        if not exists:
            body = json.dumps(
                {
                    "vectors": {
                        "size": int(vector_size),
                        "distance": "Cosine",
                    }
                }
            ).encode("utf-8")
            self._request_json("PUT", f"/collections/{self._config.collection}", body=body)

        self._vector_size = int(vector_size)
        self._initialized = True

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        import json

        if len(chunks) == 0:
            return
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError(f"Expected vectors shape (n,d) with n={len(chunks)}; got {vectors.shape}.")

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        self._ensure_collection(vector_size=int(vectors.shape[1]))

        base = self._count
        points: List[Dict[str, Any]] = []
        for i, (ch, vec) in enumerate(zip(chunks, vectors)):
            points.append(
                {
                    "id": int(base + i),
                    "vector": vec.tolist(),
                    "payload": {
                        "chunk_id": ch.chunk_id,
                        "doc_id": ch.doc_id,
                        "source_path": ch.source_path,
                        "chunk_index": int(ch.chunk_index),
                    },
                }
            )

        body = json.dumps({"points": points}).encode("utf-8")
        self._request_json(
            "PUT",
            f"/collections/{self._config.collection}/points?wait=true",
            body=body,
        )
        self._count += len(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        import json

        if top_k <= 0:
            return []
        if not self._initialized:
            return []

        q = query_vector.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)

        body = json.dumps(
            {
                "vector": q.tolist(),
                "limit": int(top_k),
                "with_payload": False,
                "with_vector": False,
            }
        ).encode("utf-8")

        data = self._request_json(
            "POST",
            f"/collections/{self._config.collection}/points/search",
            body=body,
        )

        results = data.get("result", []) if isinstance(data, dict) else []
        out: List[Tuple[int, float]] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            try:
                idx = int(r.get("id"))
                score = float(r.get("score"))
            except Exception:
                continue
            if idx < 0 or idx >= self._count:
                # Prevent stale points (e.g. if collection wasn't recreated) from crashing callers that
                # assume idx maps into the in-memory `pipeline._chunks` list.
                continue
            out.append((idx, score))
        return out
