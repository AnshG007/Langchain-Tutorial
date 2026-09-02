"""
File: faiss_vectorstore.py

Description
-----------
Demonstrates how to create and query a FAISS vector database.

FAISS is an in-memory vector database optimized for efficient similarity
search over high-dimensional embeddings.
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.helpers import print_separator, print_title

load_dotenv()


def main() -> None:
    """Demonstrates FAISS Vector Store."""

    print_title("FAISS Vector Store")

    file_path = PROJECT_ROOT / "data" / "input" / "sample.txt"

    loader = TextLoader(file_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = FAISS.from_documents(
        documents=chunks, 
        embedding=embeddings
    )

    print(f"Document Indexed: {len(chunks)}")
    print_separator()

    query1 = "Who works on the principle of neural networks?"
    query2 = "Who is the subset of AI?"
    query3 = "What is machine learning?"

    results = vector_store.similarity_search(
        query=query3,
        k=2,
    )

    print(f"Query:\n{query3}")

    print_separator()

    for index, document in enumerate(results, start=1):

        print(f"Result {index}\n")

        print(document.page_content)

        print_separator()


if __name__ == "__main__":
    main()


# =============================================================================
# Concept Summary
#
# FAISS is a high-performance vector database developed by Meta.
#
# It stores embeddings in memory and performs very fast similarity search,
# making it ideal for local RAG applications and experimentation.
#
# Unlike Chroma, FAISS focuses on fast vector search and does not provide
# built-in persistence or metadata management by default.
# =============================================================================