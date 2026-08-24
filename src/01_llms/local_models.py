"""
File: local_models.py

Description
-----------
Demonstrates how to work with locally hosted LLMs (such as Ollama)
using LangChain instead of cloud-based APIs.

Before running this script:

1. Install Ollama
2. Pull a model (example: llama3.2)
       ollama pull llama3.2
3. Start the Ollama server
       ollama serve

This example intentionally creates the model locally instead of using
the shared llm_client.py because local models typically don't require
API keys or cloud provider configuration.
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_ollama import ChatOllama

from utils.helpers import print_separator, print_title


def main() -> None:
    """
    Demonstrates interacting with a locally hosted LLM using Ollama.
    """

    print_title("Local LLM using Ollama")

    llm = chat_ollama = ChatOllama(
        model = "Llama3.2",  # Specify the model you pulled with Ollama
        temperature = 0.7,
        max_tokens = 512

    )

    prompt = """
    Explain the benefits of using locally hosted LLMs for sensitive data
    processing in exactly three concise bullet points."""

    response = llm.invoke(prompt)

    print("USER PROMPT")
    print(prompt.strip())

    print_separator()

    response = llm.invoke(prompt)

    print("Local Ollama MODEL RESPONSE")
    print(response.content)

    print_separator()

    print("Response Type:")
    print(type(response))


if __name__ == "__main__":
    main()

# Also, find and delete your model from: C:\Users\<YourUsername>\.ollama\models
    