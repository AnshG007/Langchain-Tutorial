"""
File: conversation_token_buffer.py

Description
-----------
Demonstrates ConversationTokenBufferMemory.

Unlike ConversationBufferWindowMemory, which limits memory by the number
of conversation turns, ConversationTokenBufferMemory limits memory based
on the total number of tokens.

This helps keep conversations within an LLM's context window.
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_classic.memory import ConversationTokenBufferMemory

from llm_client import get_llm
from utils.helpers import print_separator, print_title


def main() -> None:
    """Demonstrates ConversationTokenBufferMemory."""

    print_title("Conversation Token Buffer Memory")

    llm = get_llm()

    memory = ConversationTokenBufferMemory(
        llm = llm,
        max_token_limit = 50,
        return_messages = True
    )

    conversations = [
        ("Hi!", "Hello!"),
        ("My name is Ansh.", "Nice to meet you, Ansh."),
        ("I am learning LangChain.", "That's great!"),
        ("Can you explain Prompt Engineering?", "Prompt Engineering is the process of designing effective prompts."),
        ("What is RAG?", "RAG combines retrieval with generation to improve factual responses."),
        ("Explain AI Agents.", "AI Agents can reason, plan and use external tools."),
    ]

    for human, ai in conversations:
        memory.save_context(
            {"input": human},
            {"output": ai},
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