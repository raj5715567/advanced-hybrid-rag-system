from chunking.models import TextChunk
from memory import format_conversation_history
from retrieval import compress_context, expand_query


def test_query_expansion_keeps_original_and_adds_keywords():
    queries = expand_query("Who is Raj in the data engineering team?")

    assert queries[0] == "Who is Raj in the data engineering team?"
    assert "raj" in queries[1]


def test_context_compression_keeps_relevant_sentence():
    chunk = TextChunk("a", "Raj is a data engineer. The cafeteria opens at noon.", "a.txt", "txt", "file", 0, {})

    compressed = compress_context("Who is Raj?", [(chunk, 0.8)], sentences_per_chunk=1)

    assert compressed[0][0].text == "Raj is a data engineer."


def test_memory_uses_only_recent_message_content():
    history = format_conversation_history([
        {"role": "user", "content": "Who is Raj?"},
        {"role": "assistant", "content": "Raj is a data engineer.", "sources": [{"ignored": True}]},
    ])

    assert "User: Who is Raj?" in history
    assert "Assistant: Raj is a data engineer." in history
