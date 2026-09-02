"""
File: vector_retriever.py

Description
-----------
Demonstrates the basic Vector Store Retriever.

A Retriever is a higher-level abstraction over a vector database.
Instead of calling similarity_search() directly, we ask the retriever
to fetch the most relevant documents.
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
    """Demonstrates a Vector Store Retriever."""

    print_title("Vector Store Retriever")

    loader = TextLoader(
        PROJECT_ROOT / "data" / "input" / "sample.txt"
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2}
    )

    query = "What are Large Language Models?"

    print(f"Query:\n{query}")

    print_separator()

    documents = retriever.invoke(query)

    for index, document in enumerate(documents, start=1):

        print(f"Retrieved Document {index}\n")

        print(document.page_content)

        print_separator()


if __name__ == "__main__":
    main()


# =============================================================================
# Concept Summary
#
# A Retriever sits on top of a vector database.
#
# Instead of manually performing similarity search, we simply provide a
# query and the retriever returns the most relevant documents.
#
# In most RAG applications, the retriever is responsible for fetching
# context before sending it to the LLM.
# =============================================================================