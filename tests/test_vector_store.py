import numpy as np

from chunking.models import TextChunk
from embeddings.vector_store import FaissVectorStore


def test_vector_store_saves_loads_and_searches(tmp_path):
    chunks = [
        TextChunk("a", "cats", "pets.txt", "txt", "file", 0, {}),
        TextChunk("b", "cars", "vehicles.txt", "txt", "file", 0, {}),
    ]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    FaissVectorStore.from_chunks(chunks, vectors).save(tmp_path)

    results = FaissVectorStore.load(tmp_path).search(np.array([1.0, 0.0], dtype="float32"), top_k=1)

    assert results[0][0].id == "a"
    assert results[0][1] == 1.0
