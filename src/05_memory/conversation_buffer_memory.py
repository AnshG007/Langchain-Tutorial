"""
File: conversation_buffer_memory.py

Description
-----------
Demonstrates ConversationBufferMemory.

ConversationBufferMemory stores the complete conversation history.
Every user message and AI response is retained and provided as context
for future interactions.
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_classic.memory import ConversationBufferMemory

from utils.helpers import print_separator, print_title


def main() -> None:
    """Demonstrates ConversationBufferMemory."""

    print_title("Conversation Buffer Memory")

    memory = ConversationBufferMemory(return_messages=True)

    memory.save_context(
        {"input":"Hi , mt name is Ansh Gupta!"},
        {"output":"Hello Ansh! How can I assist you today?"}
    )

    memory.save_context(
        {"input": "I am learning LangChain."},
        {"output": "That's great! LangChain is an excellent framework for building LLM applications."},
    )

    memory.save_context(
        {"input": "I also want to learn RAG."},
        {"output": "RAG is one of the most important concepts in modern GenAI applications."},
    )

    print("Complete Conversation History:\n")

    history = memory.load_memory_variables({})

    for index, message in enumerate(history["history"], start=1):
        print(f"Message {index}")
        print(f"Type    : {message.type}")
        print(f"Content : {message.content}")
        print_separator()


if __name__ == "__main__":
    main()


