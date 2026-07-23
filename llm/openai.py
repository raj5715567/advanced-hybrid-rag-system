"""OpenAI Responses API adapter."""

import os

from openai import OpenAI


def generate_openai_answer(system_prompt: str, user_prompt: str, model: str) -> str:
    """Generate a grounded answer through the OpenAI Responses API."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your local .env file and restart Streamlit.")
    client = OpenAI()
    response = client.responses.create(model=model, instructions=system_prompt, input=user_prompt)
    return response.output_text.strip()

