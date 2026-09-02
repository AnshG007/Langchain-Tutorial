"""
File: similarity_search.py

Description
-----------
Demonstrates similarity search using a vector database.

Instead of matching keywords, similarity search retrieves documents whose
meaning is most similar to the user's query.
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
    """Demonstrates similarity search."""

    print_title("Similarity Search")

    loader = TextLoader(
        PROJECT_ROOT / "data" / "input" / "sample.txt"
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    query = "Explain Large Language Models."

    print(f"User Query:\n{query}")

    print_separator()

    results = vector_store.similarity_search(
        query=query,
        k=3,
    )

    for index, document in enumerate(results, start=1):

        print(f"Retrieved Chunk {index}")

        print()

        print(document.page_content)

        print_separator()


if __name__ == "__main__":
    main()


# =============================================================================
# Concept Summary
#
# Similarity Search retrieves documents based on semantic meaning rather
# than exact keyword matches.
#
# The user's query is first converted into an embedding. The vector
# database then finds the document chunks whose embeddings are closest
# to the query embedding.
#
# This is the core retrieval mechanism used in almost every RAG system.
# =============================================================================