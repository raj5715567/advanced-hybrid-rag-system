from llm.gemini import generate_gemini_answer


def test_gemini_adapter_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    try:
        generate_gemini_answer("system", "question", "gemini-3.6-flash")
    except RuntimeError as error:
        assert "GEMINI_API_KEY is missing" in str(error)
    else:
        raise AssertionError("Expected a missing-key error")
