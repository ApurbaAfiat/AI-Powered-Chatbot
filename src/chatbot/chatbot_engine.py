from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol
import os

from src.chatbot.prompt_template import build_prompt
from src.memory.conversation_memory import ConversationMemory
from src.retrieval.retriever import RetrievedChunk, Retriever

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional dependency
    ChatOpenAI = None

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency
    pipeline = None


class AnswerGenerator(Protocol):
    """Backend interface for response generation."""

    def generate(self, prompt: str) -> str:
        ...


class OpenAIChatGenerator:
    """Generate answers with OpenAI when an API key is available."""

    def __init__(self, model_name: str) -> None:
        if ChatOpenAI is None:  # pragma: no cover - optional dependency
            raise ImportError("langchain-openai is required for OpenAI generation")
        self._llm = ChatOpenAI(model=model_name, temperature=0)

    def generate(self, prompt: str) -> str:
        response = self._llm.invoke(prompt)
        return getattr(response, "content", str(response)).strip()


class LocalHuggingFaceGenerator:
    """Generate answers with a local Hugging Face text-to-text model."""

    def __init__(self, model_name: str) -> None:
        if pipeline is None:  # pragma: no cover - optional dependency
            raise ImportError("transformers is required for the local generation fallback")
        self._pipeline = pipeline("text2text-generation", model=model_name)

    def generate(self, prompt: str) -> str:
        result = self._pipeline(prompt, max_new_tokens=220, do_sample=False)
        if isinstance(result, list) and result:
            return result[0].get("generated_text", "").strip()
        return str(result).strip()


class ExtractiveFallbackGenerator:
    """Safe fallback when no language model backend is available."""

    def generate(self, prompt: str) -> str:
        marker = "Retrieved context:"
        context = prompt.split(marker, 1)[-1].strip() if marker in prompt else prompt
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        if not lines:
            return "I do not have enough information in the retrieved context to answer that."
        summary = " ".join(lines[:4])
        return summary[:700]


@dataclass
class ChatResponse:
    """Final chatbot answer with the sources used to produce it."""

    answer: str
    sources: List[str]
    retrieved_chunks: List[RetrievedChunk]


class ChatbotEngine:
    """Coordinate retrieval, memory, and answer generation."""

    def __init__(self, retriever: Retriever, memory: ConversationMemory, generator: AnswerGenerator) -> None:
        self.retriever = retriever
        self.memory = memory
        self.generator = generator

    def answer(self, question: str, top_k: int | None = None) -> ChatResponse:
        memory_augmented_question = self.memory.augment_query(question)
        results = self.retriever.retrieve(memory_augmented_question, top_k=top_k)
        context = self.retriever.build_context(results)
        memory_summary = self.memory.summary()
        prompt = build_prompt(question=question, context=context, memory_summary=memory_summary)
        answer = self.generator.generate(prompt).strip()

        sources = self._format_sources(results)
        if sources and "source:" not in answer.lower():
            answer = f"{answer}\n\nSource: {sources[0]}"

        self.memory.add_turn(question, answer, source_refs=sources)
        return ChatResponse(answer=answer, sources=sources, retrieved_chunks=results)

    def _format_sources(self, results: List[RetrievedChunk]) -> List[str]:
        formatted: List[str] = []
        seen = set()
        for result in results:
            hit = result.hit
            label = f"{hit.section} | Page {hit.page_number} | Chunk {hit.chunk_id}"
            if label not in seen:
                seen.add(label)
                formatted.append(label)
        return formatted



def build_generator(openai_model_name: str, local_model_name: str) -> AnswerGenerator:
    """Choose the best available generation backend for the environment."""

    if os.getenv("OPENAI_API_KEY", "").strip() and ChatOpenAI is not None:
        return OpenAIChatGenerator(model_name=openai_model_name)
    try:
        return LocalHuggingFaceGenerator(model_name=local_model_name)
    except Exception:
        return ExtractiveFallbackGenerator()
