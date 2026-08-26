"""
File: conversation_buffer_window.py

Description
-----------
Demonstrates ConversationBufferWindowMemory.

Unlike ConversationBufferMemory, this memory stores only the most
recent conversation messages.

This helps reduce context size while still preserving recent interactions.
"""

from pathlib import Path
import sys

import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_classic.memory import ConversationBufferWindowMemory

from utils.helpers import print_separator, print_title


def main() -> None:
    """Demonstrates ConversationBufferWindowMemory."""

    print_title("Conversation Buffer Window Memory")

    # Keep only the last 2 conversation exchanges
    memory = ConversationBufferWindowMemory(k=2, return_messages=True)

    conversations = [
        ("Hi!", "Hello!"),
        ("My name is Ansh.", "Nice to meet you, Ansh."),
        ("I am learning LangChain.", "Great choice!"),
        ("Tell me about RAG.", "RAG combines retrieval with generation."),
        ("What are AI Agents?", "AI Agents can plan, reason, and use tools."),
    ]

    for human , ai in conversations:
        memory.save_context(
            {"input": human},
            {"output": ai}
        )

    history = memory.load_memory_variables({})

    print("Conversation Stored In Memory:\n")

    for index, message in enumerate(history["history"], start=1):
        print(f"Message {index}")
        print(f"Type    : {message.type}")
        print(f"Content : {message.content}")
        print_separator()


if __name__ == "__main__":
    main()