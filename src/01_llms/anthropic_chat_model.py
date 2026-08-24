"""
File: anthropic_chat_model.py

Description
-----------
Demonstrates how to interact with Anthropic Claude models using LangChain.

This example reuses the centralized LLM configuration provided by
`llm_client.py`. To use Claude, simply configure the provider as
"anthropic" inside config.json.
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm
from utils.helpers import print_separator, print_title


def main() -> None:
    """
    Demonstrates interacting with an Anthropic Claude chat model.
    """

    print_title("Anthropic Claude Chat Model")

    print(
        "Ensure that 'config.json' is configured to use the "
        "'anthropic' provider before running this example.\n"
    )

    llm = get_llm()

    prompt = """
    Explain why Claude models are popular for enterprise AI applications
    in exactly three concise bullet points.
    """

    print("USER PROMPT")
    print(prompt.strip())

    print_separator()

    response = llm.invoke(prompt)

    print("MODEL RESPONSE")
    print(response.content)

    print_separator()

    print("Response Type:")
    print(type(response))

    print("\nAccessing only the generated text:")
    print(response.content)


if __name__ == "__main__":
    main()