"""
File: chat_history.py

Description
-----------
Demonstrates how chat messages are stored using ChatMessageHistory.

ChatMessageHistory is the foundation for conversational AI applications.
It stores all messages exchanged between the user and the AI.
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from utils.helpers import print_separator, print_title


def main() -> None:
    """Demonstrates ChatMessageHistory."""

    print_title("Chat Message History")

    chat_history = InMemoryChatMessageHistory()

    chat_history.add_message(
        HumanMessage(content = "What is Langchain?")
    )
    chat_history.add_message(
        AIMessage(content = "Langchain is a framework for building applications with LLMs.")
    )

    chat_history.add_message(
        HumanMessage(content = "What are PromptTemplates?")
    )
    chat_history.add_message(
        AIMessage(content = "PromptTemplates are reusable templates for creating prompts.")

    )

    print("Conversation History:\n")

    for index, message in enumerate(chat_history.messages, start=1):
        print(f"Message {index}")
        print(f"Type    : {message.type}")
        print(f"Content : {message.content}")
        print_separator()


if __name__ == "__main__":
    main()