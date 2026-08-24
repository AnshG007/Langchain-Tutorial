"""
File: one_shot_prompt.py

Description
-----------
Demonstrates One-Shot Prompting.

One-Shot Prompting provides the model with a single example before asking
it to perform a similar task.
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
    """Demonstrates One-Shot Prompting."""

    print_title("One-Shot Prompting")

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert sentiment analysis assistant.",
            ),
            (
                "human",
                """
Classify the sentiment as Positive, Negative, or Neutral.

Example:

Review: "The laptop is fast, lightweight, and has an amazing battery life."
Sentiment: Positive

Now classify the following review:

Review: "{review}"
""",
            ),
        ]
    )

    messages = prompt.invoke(
        {
            "review": "The phone looks premium, but its battery drains very quickly."
        }
    )

    print("Formatted Prompt:\n")

    for message in messages.messages:
        print(f"{message.type.upper()}:")
        print(message.content)
        print()

    print_separator()

    response = llm.invoke(messages)

    print("LLM Response:\n")
    print(response.text)


if __name__ == "__main__":
    main()