from __future__ import annotations

import os

from rag import Chunker, GeminiChatLLM, GeminiEmbedder, RAGPipeline

from webapp.config import AppConfig
from webapp.services.qdrant_index import QdrantCosineIndex, QdrantIndexConfig


def create_pipeline(config: AppConfig) -> RAGPipeline:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Missing API key. Set GOOGLE_API_KEY (in shell or .env).")

    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "rag_chunks").strip() or "rag_chunks"
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
    qdrant_recreate = os.getenv("QDRANT_RECREATE_COLLECTION", "1").strip() != "0"

    chunker = Chunker(
        max_tokens=config.chunk_max_tokens,
        overlap_tokens=config.chunk_overlap_tokens,
    )
    embedder = GeminiEmbedder(
        model=config.gemini_embed_model,
        output_dimensionality=None,
        batch_size=64,
        max_requests_per_minute=config.max_embed_requests_per_minute,
    )
    llm = GeminiChatLLM(
        model=config.gemini_chat_model,
        temperature=0.2,
    )

    dense_index = None
    if qdrant_url:
        dense_index = QdrantCosineIndex(
            QdrantIndexConfig(
                url=qdrant_url,
                collection=qdrant_collection,
                api_key=qdrant_api_key,
                recreate_collection_on_first_add=qdrant_recreate,
            )
        )

    return RAGPipeline(embedder=embedder, llm=llm, chunker=chunker, dense_index=dense_index)
