"""Ollama adapter for locally hosted LLMs."""

import ollama


def generate_ollama_answer(system_prompt: str, user_prompt: str, model: str) -> str:
    """Generate a grounded answer with a local Ollama model."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"].strip()

