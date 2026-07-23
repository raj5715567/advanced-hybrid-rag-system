from chunking.splitter import chunk_documents
from ingestion.models import ParsedDocument


def test_chunks_preserve_source_and_overlap():
    document = ParsedDocument(
        text="one two three four five six seven eight nine ten",
        source_name="note.txt",
        source_type="txt",
        location="entire file",
        metadata={},
    )

    chunks = chunk_documents([document], chunk_size=18, chunk_overlap=5)

    assert len(chunks) > 1
    assert all(chunk.source_name == "note.txt" for chunk in chunks)
    assert chunks[1].text.startswith(chunks[0].text[-5:].strip())
