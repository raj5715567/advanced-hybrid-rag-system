"""Grounded prompt construction for the RAG answer step."""

from chunking.models import TextChunk


SYSTEM_PROMPT = """You are a careful knowledge-base assistant. Answer using only the supplied sources.
If the sources do not contain the answer, say that clearly. Do not invent details.
Use source markers such as [1] and [2] beside claims. Keep the answer concise and helpful."""


def build_rag_prompt(question: str, results: list[tuple[TextChunk, float]], conversation_history: str = "") -> str:
    """Number retrieved chunks so the model can cite them deterministically."""
    context = "\n\n".join(
        f"[{position}] {chunk.source_name} — {chunk.location}\n{chunk.text}"
        for position, (chunk, _score) in enumerate(results, start=1)
    )
    history = f"Conversation history:\n{conversation_history}\n\n" if conversation_history else ""
    return f"""{history}Sources:
{context}

Question: {question}

Answer the question using only the sources above and include the relevant source markers."""
