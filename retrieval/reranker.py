"""Cross-encoder reranking for the hybrid retrieval candidates."""

from functools import lru_cache
from typing import Callable

from chunking.models import TextChunk


@lru_cache(maxsize=2)
def _load_reranker(model: str):
    """Load and cache the reranking model only when it is first needed."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise RuntimeError("Cross-encoder support is not installed. Run: pip install -r requirements.txt") from error
    return CrossEncoder(model)


def rerank_chunks(
    query: str,
    candidates: list[tuple[TextChunk, float]],
    top_k: int,
    model: str,
    score_pairs: Callable[[str, list[TextChunk], str], list[float]] | None = None,
) -> list[tuple[TextChunk, float]]:
    """Score each query/chunk pair and retain the most relevant chunks."""
    if not query.strip() or not candidates or top_k <= 0:
        return []

    chunks = [chunk for chunk, _ in candidates]
    if score_pairs is None:
        reranker = _load_reranker(model)
        scores = reranker.predict([(query, chunk.text) for chunk in chunks])
    else:
        scores = score_pairs(query, chunks, model)
    if len(scores) != len(chunks):
        raise ValueError("The reranker returned a score count that does not match the candidate count.")

    ranked = sorted(zip(chunks, scores), key=lambda item: (-float(item[1]), item[0].id))
    return [(chunk, float(score)) for chunk, score in ranked[:top_k]]
