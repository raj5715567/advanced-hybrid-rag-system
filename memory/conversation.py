"""Short-term conversation memory for follow-up questions."""


def format_conversation_history(messages: list[dict], max_messages: int = 6) -> str:
    """Format the latest chat turns without carrying source payloads into the prompt."""
    turns = messages[-max_messages:]
    return "\n".join(f"{item['role'].title()}: {item['content']}" for item in turns if item.get("content"))
