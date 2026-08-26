"""
File: conversation_summary_memory.py

Description
-----------
Demonstrates ConversationSummaryMemory.

Instead of storing the complete conversation, older interactions are
summarized by an LLM. This reduces token usage while preserving the
important context.
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_classic.memory import ConversationSummaryMemory

from llm_client import get_llm
from utils.helpers import print_separator, print_title


def main() -> None:
    """Demonstrates ConversationSummaryMemory."""

    print_title("Conversation Summary Memory")

    llm = get_llm()

    memory = ConversationSummaryMemory(
        llm=llm,
    )

    conversations = [
        ("Hi!", "Hello!"),
        ("My name is Ansh.", "Nice to meet you, Ansh."),
        ("I am learning LangChain.", "Great choice!"),
        ("Can you explain Prompt Engineering?", "Prompt Engineering is the process of designing effective prompts."),
        ("Tell me about RAG.", "RAG combines retrieval with generation."),
        ("What are AI Agents?", "AI Agents can reason, plan and use tools."),
    ]

    for human, ai in conversations:
        memory.save_context(
            {"input": human},
            {"output": ai},
        )

    history = memory.load_memory_variables({})

    print("Conversation Summary:\n")
    print(history["history"])

    print_separator()


if __name__ == "__main__":
    main()