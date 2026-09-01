"""
File: recursive_splitter.py

Description
-----------
Demonstrates RecursiveCharacterTextSplitter.

This is the most commonly used text splitter in LangChain because it
attempts to preserve the semantic meaning of the text while creating
smaller chunks.

It recursively splits text using a hierarchy of separators such as:

Paragraphs -> Lines -> Spaces -> Characters
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.helpers import print_separator, print_title


def main() -> None:
    """Demonstrates RecursiveCharacterTextSplitter."""

    print_title("Recursive Character Text Splitter")

    file_path = PROJECT_ROOT / "data" / "input" / "sample.txt"

    loader = TextLoader(file_path)

    documents = loader.load()

    print(f"Original Documents: {len(documents)}")
    print_separator()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    print(f"Total Chunks Created: {len(chunks)}")
    print_separator()

    for index, chunk in enumerate(chunks, start=1):
        print(f"Chunk {index}")
        print(f"Characters : {len(chunk.page_content)}")
        print()
        print(chunk.page_content)

        print_separator()


if __name__ == "__main__":
    main()


# =============================================================================
# Concept Summary
#
# RecursiveCharacterTextSplitter is the most commonly used text splitter
# in LangChain because it tries to preserve the meaning of the text.
#
# Instead of splitting strictly by characters, it first tries to split
# using paragraphs, then lines, then spaces, and finally individual
# characters if necessary.
#
# It is the default choice for most RAG applications.
# =============================================================================