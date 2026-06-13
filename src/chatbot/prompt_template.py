from __future__ import annotations

SYSTEM_PROMPT = """You are a financial policy assistant.
Answer only from the supplied policy context.
Keep the response concise, factual, and helpful.
Always include a short source reference line using the chunk metadata.
If the context does not contain enough information, say so clearly.
"""


def build_prompt(question: str, context: str, memory_summary: str = "") -> str:
    """Render the final prompt passed to the generation backend."""

    parts = [SYSTEM_PROMPT.strip(), f"Question:\n{question.strip()}"]
    if memory_summary.strip():
        parts.append(f"Conversation memory:\n{memory_summary.strip()}")
    parts.append(f"Retrieved context:\n{context.strip()}")
    parts.append(
        "Answer format:\n"
        "- concise answer\n"
        "- supporting context\n"
        "- Source: <section> | Page <page> | Chunk <chunk_id>"
    )
    return "\n\n".join(parts)
