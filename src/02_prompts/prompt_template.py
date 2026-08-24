"""
File: prompt_template.py

Description
-----------
Demonstrates how to create reusable prompts using PromptTemplate.

PromptTemplate allows us to define placeholders that can be dynamically
filled at runtime, making prompts reusable and easier to maintain.
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.prompts import PromptTemplate

from llm_client import get_llm
from utils.helpers import print_separator, print_title

def main():
    print_title("PromptTemplate Example")

    llm = get_llm()

    prompt_template = PromptTemplate(
        template="""
        you are the expert {profession}
        Explain the concept of {topic}
        keep the explanation under {word_limit} words
        """,
        input_variables=["profession", "topic", "word_limit"]
    )

    prompt = prompt_template.invoke(
        {
        "profession":"AI Engineer",
        "topic":"Vector Databases",
        "word_limit":100,
        },
    )

    print_title("PromptTemplate Output")
    print(prompt.text)
    print_separator()


    response = llm.invoke(prompt)
    print_separator()

    print_title("LLM RESPONSE")
    print(response.text)
    print_separator()

if __name__ == "__main__":
    main()