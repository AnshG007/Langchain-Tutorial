"""
File: pdf_loader.py

Description
-----------
Demonstrates how to load PDF documents using PyPDFLoader.

PDFs are one of the most common data sources used in Retrieval-Augmented
Generation (RAG) applications.
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import PyPDFLoader

from utils.helpers import print_separator, print_title


def main() -> None:
    """Demonstrates PyPDFLoader."""

    print_title("PDF Loader")

    file_path = PROJECT_ROOT / "data" / "input" / "sample.pdf"

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    print(f"Total Pages Loaded : {len(documents)}")

    print_separator()

    first_page = documents[0]

    print("Page Metadata:\n")
    print(first_page.metadata)

    print_separator()

    print("First Page Content:\n")
    print(first_page.page_content)


if __name__ == "__main__":
    main()


# =============================================================================
# Concept Summary
#
# PyPDFLoader reads PDF documents and converts each page into a
# LangChain Document object.
#
# Each page becomes an individual document containing:
#
# 1. page_content → Text extracted from the page
# 2. metadata     → Source file and page number
#
# These documents are later split into chunks for RAG.
# =============================================================================