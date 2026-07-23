from chunking.models import TextChunk
from retrieval.bm25 import retrieve_bm25
from retrieval.hybrid import reciprocal_rank_fusion


def _chunk(chunk_id: str, text: str) -> TextChunk:
    return TextChunk(chunk_id, text, "source.txt", "txt", "file", 0, {})


def test_bm25_prioritises_exact_keyword_matches():
    raj = _chunk("raj", "Raj builds enterprise data pipelines.")
    other = _chunk("other", "A recipe for vegetable soup.")

    results = retrieve_bm25("Who is Raj?", [other, raj], top_k=2)

    assert results[0][0].id == "raj"


def test_rrf_rewards_chunks_found_by_both_retrievers():
    shared = _chunk("shared", "shared")
    dense_only = _chunk("dense", "dense")
    bm25_only = _chunk("bm25", "bm25")

    results = reciprocal_rank_fusion(
        [[(dense_only, 0.9), (shared, 0.8)], [(bm25_only, 5.0), (shared, 4.0)]], top_k=3
    )

    assert results[0][0].id == "shared"
