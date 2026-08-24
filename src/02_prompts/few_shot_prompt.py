"""
File: few_shot_prompt.py

Description
-----------
Demonstrates Few-Shot Prompting.

Few-Shot Prompting provides multiple examples to the model before asking
it to solve a similar task. This technique helps the model understand
the expected output format and improves consistency.
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
    """Demonstrates Few-Shot Prompting."""

    print_title("Few-Shot Prompting")

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an AI assistant that categorizes customer support tickets.

                Classify each ticket into exactly one category from the following:

                - Billing
                - Technical Issue
                - Account Management
                - Feature Request
                - General Inquiry

                Respond ONLY with the category name.
                """,
            ),
            (
                "human",
                """
                Example 1

                Ticket:
                I was charged twice for my monthly subscription.

                Category:
                Billing

                ----------------------------------------

                Example 2

                Ticket:
                I forgot my password and can't log into my account.

                Category:
                Account Management

                ----------------------------------------

                Example 3

                Ticket:
                The mobile app crashes every time I upload a photo.

                Category:
                Technical Issue

                ----------------------------------------

                Now classify this ticket:

                Ticket:
                {ticket}

                Category:
                """,
            ),
        ]
    )

    messages = prompt.invoke(
        {
            "ticket": (
                "It would be great if your platform could support "
                "dark mode for the dashboard."
            )
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