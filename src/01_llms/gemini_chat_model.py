"""
File: gemini_chat_model.py

Description
-----------
Demonstrates how to interact with Google's Gemini chat models using LangChain.

In this project, the model is NOT created directly inside this file.
Instead, we reuse the centralized LLM client defined in `llm_client.py`.

To run this example:

1. Open config.json
2. Change the provider to "gemini"
3. Ensure your Gemini API key is configured in the .env file
4. Run this script

Notice that the application code remains unchanged. Only the configured
provider changes, which highlights one of LangChain's biggest strengths:
a unified interface across multiple LLM providers.
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# This allows the script to be executed independently from anywhere.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm
from utils.helpers import print_separator, print_title


def main() -> None:
    """
    Demonstrates interacting with a Gemini chat model using LangChain.
    """

    print_title("Google Gemini Chat Model")

    print(
        "This lesson assumes that 'config.json' is configured to use "
        "the Gemini provider.\n"
    )

    llm = get_llm()

    prompt = """
    Explain the benefits of using LangChain with Gemini in exactly three concise bullet points.
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