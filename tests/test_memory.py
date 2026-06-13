from src.memory.conversation_memory import ConversationMemory


def test_memory_tracks_previous_turns_and_topic() -> None:
    memory = ConversationMemory(max_turns=3)
    memory.add_turn("What are the financial objectives?", "They include balanced budget and low debt.")
    memory.add_turn("What about debt?", "Debt should remain low.")

    assert memory.previous_question == "What about debt?"
    assert "Previous answer" in memory.summary()
    assert "debt" in memory.current_topic
    assert "Related topic" in memory.augment_query("What about debt?")
