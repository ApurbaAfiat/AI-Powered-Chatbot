from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
import re
from typing import List, Optional


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "about",
    "what",
    "which",
    "who",
    "how",
    "why",
    "when",
    "do",
    "does",
    "did",
    "this",
    "that",
    "it",
    "its",
    "your",
    "our",
    "their",
    "them",
    "they",
}


@dataclass
class ConversationTurn:
    """A single user/assistant exchange."""

    user_question: str
    assistant_answer: str
    topic: str
    source_refs: List[str] = field(default_factory=list)


class ConversationMemory:
    """Keep the recent dialogue state for follow-up questions."""

    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max_turns
        self._turns: List[ConversationTurn] = []

    @property
    def turns(self) -> List[ConversationTurn]:
        return list(self._turns)

    @property
    def previous_question(self) -> str:
        return self._turns[-1].user_question if self._turns else ""

    @property
    def previous_answer(self) -> str:
        return self._turns[-1].assistant_answer if self._turns else ""

    @property
    def current_topic(self) -> str:
        return self._turns[-1].topic if self._turns else ""

    def add_turn(self, user_question: str, assistant_answer: str, source_refs: Optional[List[str]] = None) -> None:
        topic = self._infer_topic(user_question)
        self._turns.append(
            ConversationTurn(
                user_question=user_question.strip(),
                assistant_answer=assistant_answer.strip(),
                topic=topic,
                source_refs=source_refs or [],
            )
        )
        self._turns = self._turns[-self.max_turns :]

    def summary(self) -> str:
        if not self._turns:
            return ""
        latest = self._turns[-1]
        parts = [f"Previous question: {latest.user_question}", f"Previous answer: {latest.assistant_answer[:220]}"]
        if latest.topic:
            parts.append(f"Current topic: {latest.topic}")
        return " | ".join(parts)

    def augment_query(self, query: str) -> str:
        summary = self.summary()
        if not summary:
            return query
        short_follow_up = len(query.split()) <= 6
        if short_follow_up and self.current_topic:
            return f"{query}\nRelated topic: {self.current_topic}\nMemory: {summary}"
        return f"{query}\nMemory: {summary}"

    def reset(self) -> None:
        self._turns.clear()

    def _infer_topic(self, text: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9%.-]+", text.lower())
        tokens = [token for token in tokens if token not in _STOPWORDS and len(token) > 2]
        if not tokens:
            return ""
        topic_words = [word for word, _ in Counter(tokens).most_common(3)]
        return " ".join(topic_words)
