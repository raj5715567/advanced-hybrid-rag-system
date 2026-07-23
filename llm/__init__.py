"""LLM provider components."""

from llm.gemini import generate_gemini_answer
from llm.ollama import generate_ollama_answer
from llm.openai import generate_openai_answer
from llm.prompt import SYSTEM_PROMPT, build_rag_prompt


def generate_answer(provider: str, system_prompt: str, user_prompt: str, model: str) -> str:
    """Route an answer request to the selected configured provider."""
    provider = provider.strip().lower()
    if provider == "gemini":
        return generate_gemini_answer(system_prompt, user_prompt, model)
    if provider == "openai":
        return generate_openai_answer(system_prompt, user_prompt, model)
    if provider == "ollama":
        return generate_ollama_answer(system_prompt, user_prompt, model)
    raise ValueError(f"Unsupported LLM provider: {provider}. Choose gemini, ollama, or openai.")


__all__ = ["SYSTEM_PROMPT", "build_rag_prompt", "generate_answer"]
