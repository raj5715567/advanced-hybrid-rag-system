"""Dense retrieval over the persistent FAISS index."""

from pathlib import Path

from chunking.models import TextChunk
from embeddings import FaissVectorStore, embed_texts


def retrieve_dense(
    query: str, index_directory: Path, embedding_model: str, top_k: int
) -> list[tuple[TextChunk, float]]:
    """Embed a question and return its most semantically similar chunks."""
    if not query.strip():
        return []
    store = FaissVectorStore.load(index_directory)
    query_vector = embed_texts([query], embedding_model)[0]
    return store.search(query_vector, top_k)

