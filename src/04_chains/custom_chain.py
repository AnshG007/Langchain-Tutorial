"""
File: custom_chain.py

Description
-----------
Demonstrates how to build a custom multi-step workflow by combining
multiple LangChain components using LCEL.

Workflow:

Topic
   ↓
Generate Blog Title
   ↓
Generate Blog Outline
   ↓
Generate LinkedIn Post
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from llm_client import get_llm
from utils.helpers import print_separator, print_title


def main() -> None:
    """Demonstrates a custom multi-step workflow."""

    print_title("Custom Chain")

    llm = get_llm()

    # --------------------------------------------------------------
    # Step 1: Generate Blog Title
    # --------------------------------------------------------------
    title_chain = (
        PromptTemplate.from_template(
            "Generate a catchy blog title about {topic}."
        )
        | llm
        | StrOutputParser()
    )

    # --------------------------------------------------------------
    # Step 2: Generate Blog Outline
    # --------------------------------------------------------------
    outline_chain = (
        PromptTemplate.from_template(
            """
            Create a detailed blog outline for the following title.

            Title:
            {title}
            """
        )
        | llm
        | StrOutputParser()
    )

    # --------------------------------------------------------------
    # Step 3: Generate LinkedIn Post
    # --------------------------------------------------------------
    linkedin_chain = (
        PromptTemplate.from_template(
            """
        Write a professional LinkedIn post to promote the following blog.

        Title:
        {title}

        Outline:
        {outline}
        """
        )
        | llm
        | StrOutputParser())

    # --------------------------------------------------------------
    # Custom Workflow
    # --------------------------------------------------------------
    def blog_generation_workflow(topic: str) -> dict:
        """Executes a multi-step content generation workflow."""

        title = title_chain.invoke({"topic": topic})

        outline = outline_chain.invoke({"title": title})

        linkedin_post = linkedin_chain.invoke(
            {
                "title": title,
                "outline": outline,
            }
        )

        return {
            "title": title,
            "outline": outline,
            "linkedin_post": linkedin_post,
        }

    custom_chain = RunnableLambda(blog_generation_workflow)

    topic = "AI Agents"

    print(f"Topic:\n{topic}")

    print_separator()

    result = custom_chain.invoke(topic)

    print("Generated Title:\n")
    print(result["title"])

    print_separator()

    print("Generated Outline:\n")
    print(result["outline"])

    print_separator()

    print("LinkedIn Post:\n")
    print(result["linkedin_post"])


if __name__ == "__main__":
    main()