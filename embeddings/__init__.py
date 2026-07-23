"""Embedding and vector-store components."""

from embeddings.embedding_model import embed_texts
from embeddings.vector_store import FaissVectorStore

__all__ = ["embed_texts", "FaissVectorStore"]
