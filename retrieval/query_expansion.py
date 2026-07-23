"""Local query expansion and context compression utilities."""

import re
from dataclasses import replace

from chunking.models import TextChunk


_STOP_WORDS = {"about", "could", "does", "from", "have", "into", "that", "the", "this", "what", "when", "where", "which", "who", "with", "would", "your"}


def expand_query(query: str) -> list[str]:
    """Create a compact keyword variant without an extra LLM API call."""
    cleaned = query.strip()
    keywords = [token for token in re.findall(r"\w+", cleaned.lower()) if len(token) > 2 and token not in _STOP_WORDS]
    keyword_query = " ".join(keywords)
    return [cleaned] if not keyword_query or keyword_query.lower() == cleaned.lower() else [cleaned, keyword_query]


def compress_context(query: str, results: list[tuple[TextChunk, float]], sentences_per_chunk: int = 3) -> list[tuple[TextChunk, float]]:
    """Keep the most query-relevant sentences from each retrieved chunk."""
    query_terms = set(re.findall(r"\w+", query.lower()))
    compressed: list[tuple[TextChunk, float]] = []
    for chunk, score in results:
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", chunk.text) if sentence.strip()]
        ranked = sorted(
            enumerate(sentences),
            key=lambda item: (-len(query_terms.intersection(re.findall(r"\w+", item[1].lower()))), item[0]),
        )[:sentences_per_chunk]
        selected = " ".join(sentence for _, sentence in sorted(ranked)) or chunk.text
        compressed.append((replace(chunk, text=selected), score))
    return compressed
