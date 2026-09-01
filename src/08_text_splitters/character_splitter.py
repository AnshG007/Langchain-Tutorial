"""
File: character_splitter.py

Description
-----------
Demonstrates CharacterTextSplitter.

CharacterTextSplitter splits text using a specified separator while
respecting the configured chunk size and chunk overlap.

Unlike RecursiveCharacterTextSplitter, it does not recursively search
for better splitting points.
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
from langchain_text_splitters import CharacterTextSplitter

from utils.helpers import print_separator, print_title


def main() -> None:
    """Demonstrates CharacterTextSplitter."""

    print_title("Character Text Splitter")

    file_path = PROJECT_ROOT / "data" / "input" / "sample.txt"

    loader = TextLoader(file_path)
    documents = loader.load()

    print(f"Original Documents : {len(documents)}")

    print_separator()

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    print(f"Chunks Created : {len(chunks)}")

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
# CharacterTextSplitter splits text using a fixed separator (such as
# newline, space, or comma).
#
# Unlike RecursiveCharacterTextSplitter, it does not try multiple
# separators to preserve the structure of the document.
#
# It is suitable for simple documents where the separator is already
# well-defined.
# =============================================================================