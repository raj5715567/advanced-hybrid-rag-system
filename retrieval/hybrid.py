"""Hybrid dense and lexical retrieval using reciprocal-rank fusion."""

from collections.abc import Sequence
from pathlib import Path

from chunking.models import TextChunk
from embeddings import FaissVectorStore, embed_texts
from retrieval.bm25 import retrieve_bm25


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[TextChunk, float]]], top_k: int, rank_constant: int = 60
) -> list[tuple[TextChunk, float]]:
    """Combine ranked result lists without comparing their incompatible scores."""
    if top_k <= 0:
        return []

    fused_scores: dict[str, float] = {}
    chunks_by_id: dict[str, TextChunk] = {}
    for ranked_list in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked_list, start=1):
            chunks_by_id[chunk.id] = chunk
            fused_scores[chunk.id] = fused_scores.get(chunk.id, 0.0) + 1 / (rank_constant + rank)

    ordered_ids = sorted(fused_scores, key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id))[:top_k]
    return [(chunks_by_id[chunk_id], fused_scores[chunk_id]) for chunk_id in ordered_ids]


def retrieve_hybrid(
    query: str,
    index_directory: Path,
    embedding_model: str,
    dense_top_k: int,
    bm25_top_k: int,
    final_top_k: int,
) -> list[tuple[TextChunk, float]]:
    """Retrieve FAISS and BM25 candidates, then fuse them with RRF."""
    if not query.strip():
        return []

    store = FaissVectorStore.load(index_directory)
    query_vector = embed_texts([query], embedding_model)[0]
    dense_results = store.search(query_vector, dense_top_k)
    bm25_results = retrieve_bm25(query, store.chunks, bm25_top_k)
    return reciprocal_rank_fusion((dense_results, bm25_results), final_top_k)
