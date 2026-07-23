"""Gemini API adapter."""

import os


def generate_gemini_answer(system_prompt: str, user_prompt: str, model: str) -> str:
    """Generate a grounded answer through the Google Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your local .env file and restart Streamlit.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise RuntimeError("Gemini support is not installed. Run: pip install -r requirements.txt") from error

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    if not response.text:
        raise RuntimeError("Gemini returned no text for this request.")
    return response.text.strip()

