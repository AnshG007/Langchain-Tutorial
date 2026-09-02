"""
File: contextual_compression.py

Description
-----------
Demonstrates Contextual Compression Retriever.

After retrieving documents, an LLM filters out irrelevant information
and keeps only the content needed to answer the user's question.
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
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (LLMChainExtractor,)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_client import get_llm
from utils.helpers import print_separator, print_title

load_dotenv()


def main() -> None:
    """Demonstrates Contextual Compression Retriever."""

    print_title("Contextual Compression Retriever")

    loader = TextLoader(
        PROJECT_ROOT / "data" / "input" / "sample.txt"
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embed_model,
    )

    base_retriever = vector_store.as_retriever()
    compressor = LLMChainExtractor.from_llm(get_llm())

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    query = "What are Large Language Models?"

    print(f"Query:\n{query}")

    print_separator()

    documents = retriever.invoke(query)

    print(f"Retrieved {len(documents)} Compressed Documents\n")

    print_separator()

    for index, document in enumerate(documents, start=1):
        print(f"Document {index}\n")
        print(document.page_content)
        print_separator()


if __name__ == "__main__":
    main()


# =============================================================================
# Concept Summary
#
# Contextual Compression retrieves documents first and then removes
# irrelevant information using an LLM.
#
# Instead of sending entire document chunks to the LLM, only the
# most relevant content is retained.
#
# This helps reduce token usage and improves answer quality.
# =============================================================================
