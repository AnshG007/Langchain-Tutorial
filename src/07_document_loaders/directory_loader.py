"""
File: directory_loader.py

Description
-----------
Demonstrates how to load multiple documents from a directory using
DirectoryLoader.

DirectoryLoader automatically discovers files inside a directory and
loads them using the specified loader class.
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)

from utils.helpers import print_separator, print_title


def main() -> None:
    """Demonstrates DirectoryLoader."""

    print_title("Directory Loader")

    directory = PROJECT_ROOT / "data" / "input"

    loader = DirectoryLoader(
        path=directory,
        glob= "**/*.txt",
        loader_cls=TextLoader,
    )

    documents = loader.load()

    print(f"Total Documents Loaded: {len(documents)}")
    print_separator()

    for index , document in enumerate(documents , start = 1):
        print(f"Document {index}")
        print(f"Source: {document.metadata['source']}")
        print()
        print(f"Content:\n{document.page_content[:200]}...")  # Print first 200 characters
        print_separator()

if __name__  == "__main__":
    main()

# =============================================================================
# Concept Summary
#
# DirectoryLoader automatically loads multiple files from a folder.
#
# Instead of loading one document at a time, it scans the directory,
# finds matching files, and converts each into a LangChain Document.
#
# This is commonly used when building RAG systems over hundreds or
# thousands of documents.
# =============================================================================