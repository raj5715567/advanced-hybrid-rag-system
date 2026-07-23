"""Lexical BM25 retrieval over the chunks stored with the FAISS index."""

import re

from rank_bm25 import BM25Plus

from chunking.models import TextChunk


def tokenize(text: str) -> list[str]:
    """Return simple, case-insensitive word tokens for keyword retrieval."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def retrieve_bm25(query: str, chunks: list[TextChunk], top_k: int) -> list[tuple[TextChunk, float]]:
    """Rank chunks by lexical BM25 relevance for a query."""
    if not query.strip() or not chunks or top_k <= 0:
        return []

    query_tokens = tokenize(query)
    corpus = [tokenize(chunk.text) for chunk in chunks]
    matching_indices = [
        index for index, document_tokens in enumerate(corpus) if set(query_tokens).intersection(document_tokens)
    ]
    if not matching_indices:
        return []

    # BM25Plus retains meaningful term weights for very small document sets,
    # where BM25Okapi can assign zero IDF to an otherwise exact match.
    scores = BM25Plus(corpus).get_scores(query_tokens)
    ranked_indices = sorted(matching_indices, key=lambda index: (-scores[index], index))[:top_k]
    return [(chunks[index], float(scores[index])) for index in ranked_indices]
