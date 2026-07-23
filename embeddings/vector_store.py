"""Persistent local FAISS vector index for retrieval-ready chunks."""

import json
from pathlib import Path

import faiss
import numpy as np

from chunking.models import TextChunk


INDEX_FILENAME = "chunks.faiss"
METADATA_FILENAME = "chunks_metadata.json"


class FaissVectorStore:
    """A small, transparent FAISS cosine-similarity index."""

    def __init__(self, index: faiss.Index, chunks: list[TextChunk]):
        self.index = index
        self.chunks = chunks

    @classmethod
    def from_chunks(cls, chunks: list[TextChunk], embeddings: np.ndarray) -> "FaissVectorStore":
        """Build an in-memory inner-product index from normalized vectors."""
        if not chunks:
            raise ValueError("Cannot build an index without chunks.")
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk must have exactly one embedding.")
        if embeddings.ndim != 2 or embeddings.shape[1] == 0:
            raise ValueError("Embeddings must be a non-empty two-dimensional array.")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.ascontiguousarray(embeddings, dtype="float32"))
        return cls(index, chunks)

    def save(self, directory: Path) -> None:
        """Persist the FAISS vectors and corresponding chunk metadata.

        FAISS's native Windows file writer can fail on Unicode paths (for
        example, a OneDrive folder with Vietnamese characters). Serializing to
        bytes lets Python handle the filesystem path safely instead.
        """
        directory.mkdir(parents=True, exist_ok=True)
        (directory / INDEX_FILENAME).write_bytes(faiss.serialize_index(self.index).tobytes())
        (directory / METADATA_FILENAME).write_text(
            json.dumps([chunk.to_dict() for chunk in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "FaissVectorStore":
        """Load a previously saved index and its aligned chunk metadata."""
        index_path = directory / INDEX_FILENAME
        metadata_path = directory / METADATA_FILENAME
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("No saved FAISS index found. Build the index first.")
        chunks = [TextChunk(**item) for item in json.loads(metadata_path.read_text(encoding="utf-8"))]
        index_bytes = np.frombuffer(index_path.read_bytes(), dtype="uint8")
        index = faiss.deserialize_index(index_bytes)
        if index.ntotal != len(chunks):
            raise ValueError("Saved FAISS index and chunk metadata do not match.")
        return cls(index, chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[TextChunk, float]]:
        """Return the most similar chunks and their cosine-similarity scores."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.shape[1] != self.index.d:
            raise ValueError("The query embedding dimension does not match this index.")
        scores, indices = self.index.search(np.ascontiguousarray(query_embedding, dtype="float32"), min(top_k, len(self.chunks)))
        return [
            (self.chunks[index], float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]
