"""
File: web_loader.py

Description
-----------
Demonstrates how to load content from a web page using WebBaseLoader.

This allows LangChain to retrieve publicly available web content
and convert it into Document objects for downstream RAG applications.
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import WebBaseLoader

from utils.helpers import print_separator, print_title


def main() -> None:
    """Demonstrates WebBaseLoader."""

    print_title("Web Loader")

    loader = WebBaseLoader(
        web_paths=(
            "https://python.langchain.com/docs/introduction/",
        )
    )

    documents = loader.load()

    print(f"Total Pages Loaded: {len(documents)}")

    print_separator()

    document = documents[0]

    print("First Page Metadata:\n")
    print(document.metadata)

    print_separator()

    print("Document Content (First 1000 Characters):\n")
    print(document.page_content[:1000])
if __name__ == "__main__":
    main()

# =============================================================================
# Concept Summary
#
# WebBaseLoader fetches content directly from web pages and converts
# it into LangChain Document objects.
#
# This is useful when building RAG applications over websites,
# documentation, blogs, or knowledge bases.
#
# After loading, the documents can be chunked, embedded, and stored
# in a vector database.
# =============================================================================