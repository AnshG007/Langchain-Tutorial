"""
File: chat_prompt_template.py

Description
-----------
Demonstrates how to use ChatPromptTemplate for chat-based language models.

Unlike PromptTemplate, ChatPromptTemplate allows us to structure prompts
using different message roles such as System, Human, and AI.
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.prompts import ChatPromptTemplate

from llm_client import get_llm
from utils.helpers import print_separator, print_title


def main() -> None:
    """Demonstrates ChatPromptTemplate."""

    print_title("ChatPromptTemplate")

    llm = get_llm()

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an experienced {profession} who explains concepts in a simple and beginner-friendly way.",
            ),
            (
                "human",
                "Explain the concept of {topic} in less than {word_limit} words.",
            ),
            (
                "ai",
                "Sure! I'll explain it in a clear and concise manner.",
            ),
        ]
    )

    messages = chat_prompt.invoke(
        {
            "profession": "AI Engineer",
            "topic": "Vector Databases",
            "word_limit": 100,
        }
    )
    
    print("Formatted Messages:\n")
    for message in messages.messages:
        print(f"{message.type.upper()}")
        print(message.text)
        print()

    print_separator()

    response = llm.invoke(messages)

    print_title("LLM RESPONSE")
    print(response.text)


if __name__ == "__main__":
    main()