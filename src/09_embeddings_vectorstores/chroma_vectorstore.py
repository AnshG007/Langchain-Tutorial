
"""
File: chroma_vectorstore.py

Description
-----------
Demonstrates how to create and query a Chroma vector database.

The document is loaded, split into chunks, converted into embeddings,
and stored inside Chroma.
"""

import os
import shutil
from pathlib import Path
import sys

# Disable Chroma telemetry to prevent hanging issues during vector additions
os.environ["ANONYMIZED_TELEMETRY"] = "False"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.helpers import print_separator, print_title

load_dotenv()


def main() -> None:
    print_title("Chroma Vector Store")

    # 1. Load and split the document
    file_path = PROJECT_ROOT / "data" / "input" / "sample.txt"
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Chunks Created : {len(chunks)}")
    print_separator()

    # 2. Setup Embedding Model & Database Path
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chroma_db_path = PROJECT_ROOT / "data" / "chroma_db"

    if chroma_db_path.exists():
        shutil.rmtree(chroma_db_path)

    # 3. Initialize & Populate Vector Store in one step
    print("Adding documents to Chroma...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="langchain_demo",
        persist_directory=str(chroma_db_path),
    )
    print("Documents successfully stored.")
    print_separator()

    # 4. Verify Document Count
    print(f"Document Count in Collection : {vector_store._collection.count()}")
    print_separator()

    # 5. Run Similarity Search
    query = "What is Generative AI?"
    print(f"Query: {query}")
    print("Running similarity search...")

    results = vector_store.similarity_search(query=query, k=2)
    print(f"Retrieved {len(results)} document(s).\n")
    print_separator()

    for idx, doc in enumerate(results, start=1):
        print(f"Result {idx}:\n{doc.page_content}\n")
        print_separator()


if __name__ == "__main__":
    main()


# =============================================================================
# Concept Summary
#
# Chroma is a vector database that stores embeddings.
#
# After converting text into vectors, Chroma enables semantic similarity
# search instead of traditional keyword matching.
#
# It is one of the most popular vector databases for local RAG projects.
# =============================================================================