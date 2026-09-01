"""
File: markdown_splitter.py

Description
-----------
Demonstrates MarkdownHeaderTextSplitter.

Unlike other text splitters, this splitter preserves the structure
of Markdown documents by creating chunks based on headings.
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_text_splitters import MarkdownHeaderTextSplitter

from utils.helpers import print_separator, print_title


def main() -> None:
    """Demonstrates MarkdownHeaderTextSplitter."""

    print_title("Markdown Header Text Splitter")

    file_path = PROJECT_ROOT / "data" / "input" / "langchain_notes.md"

    with open(file_path, "r", encoding="utf-8") as file:
        markdown_content = file.read()

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#" , "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    )

    chunks = splitter.split_text(markdown_content)

    print(f"Chunks Created : {len(chunks)}")
    print_separator()

    for index , chunk in enumerate(chunks, start=1):
        print(f"Chunk ->> {index}")
        print("\nMetadata:")
        print(chunk.metadata)

        print("\nContent:")
        print(chunk.page_content)

        print_separator()


if __name__ == "__main__":
    main()


# =============================================================================
# Concept Summary
#
# MarkdownHeaderTextSplitter creates chunks based on Markdown headings.
#
# Instead of splitting text purely by size, it preserves the document
# hierarchy such as sections and sub-sections.
#
# This is especially useful when working with technical documentation,
# README files, knowledge bases, and project documentation.
# =============================================================================