from __future__ import annotations

import os

from rag import Chunker, GeminiChatLLM, GeminiEmbedder, RAGPipeline

from webapp.config import AppConfig


def create_pipeline(config: AppConfig) -> RAGPipeline:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Missing API key. Set GOOGLE_API_KEY (in shell or .env).")

    chunker = Chunker(
        max_tokens=config.chunk_max_tokens,
        overlap_tokens=config.chunk_overlap_tokens,
    )
    embedder = GeminiEmbedder(
        model=config.gemini_embed_model,
        output_dimensionality=None,
        batch_size=64,
    )
    llm = GeminiChatLLM(
        model=config.gemini_chat_model,
        temperature=0.2,
    )
    return RAGPipeline(embedder=embedder, llm=llm, chunker=chunker)
