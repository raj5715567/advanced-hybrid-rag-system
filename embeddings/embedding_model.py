"""Local embedding-model loading and encoding."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Load a Sentence Transformer once per Streamlit server process."""
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """Create unit-normalized vectors suitable for cosine-similarity search."""
    if not texts:
        return np.empty((0, 0), dtype="float32")
    model = get_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.ascontiguousarray(embeddings, dtype="float32")

