from __future__ import annotations

from rag import RAGPipeline, rrf_fuse

from webapp.config import QuerySettings


def fast_retrieve(pipeline: RAGPipeline, query: str, settings: QuerySettings) -> dict:
    queries = pipeline._multi_query_variants(query, use_llm=False, n=4)

    dense_lists = []
    sparse_lists = []
    for query_variant in queries:
        dense_lists.append(pipeline._retrieve_dense(query_variant, settings.dense_top_k))
        sparse_lists.append(pipeline._retrieve_bm25(query_variant, settings.sparse_top_k))

    fused = rrf_fuse([*dense_lists, *sparse_lists], k=60, top_k=settings.fuse_top_k)

    candidates = []
    fused_scores = []
    for idx, score in fused:
        candidates.append(pipeline._chunks[idx])
        fused_scores.append(score)

    matches = [
        {
            "rank": i + 1,
            "chunk_id": chunk.chunk_id,
            "source_path": chunk.source_path,
            "chunk_index": chunk.chunk_index,
            "rrf_score": score,
            "text_preview": chunk.text[:800],
        }
        for i, (chunk, score) in enumerate(
            zip(candidates[: settings.retrieve_top_k], fused_scores[: settings.retrieve_top_k])
        )
    ]

    return {
        "matches": matches,
        "queries_used": queries,
        "top_fused_score": fused_scores[0] if fused_scores else None,
    }
