from chunking.models import TextChunk
from retrieval.reranker import rerank_chunks


def _chunk(chunk_id: str, text: str) -> TextChunk:
    return TextChunk(chunk_id, text, "source.txt", "txt", "file", 0, {})


def test_reranker_uses_cross_encoder_scores_not_original_rank():
    weak = _chunk("weak", "Raj is a chef.")
    strong = _chunk("strong", "Raj is a data engineer.")

    results = rerank_chunks(
        "Who is Raj?",
        [(weak, 0.9), (strong, 0.8)],
        top_k=1,
        model="test-model",
        score_pairs=lambda query, chunks, model: [0.1, 0.95],
    )

    assert results == [(strong, 0.95)]
