import json

from ingestion.models import ParsedDocument
from ingestion.parser import save_processed_documents


def test_save_processed_documents(tmp_path):
    document = ParsedDocument("Example text", "note.txt", "txt", "entire file", {})
    output = save_processed_documents([document], tmp_path / "parsed.json")

    assert json.loads(output.read_text(encoding="utf-8"))[0]["source_name"] == "note.txt"
