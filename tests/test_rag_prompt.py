from chunking.models import TextChunk
from llm.prompt import build_rag_prompt


def test_rag_prompt_numbers_and_includes_sources():
    chunk = TextChunk("chunk-1", "Raj is a data engineer.", "resume.pdf", "pdf", "page 1", 0, {})

    prompt = build_rag_prompt("Who is Raj?", [(chunk, 0.9)])

    assert "[1] resume.pdf — page 1" in prompt
    assert "Raj is a data engineer." in prompt
    assert "Who is Raj?" in prompt
