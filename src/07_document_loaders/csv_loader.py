"""
File: csv_loader.py

Description
-----------
Demonstrates how to load CSV files using CSVLoader.

CSV files are commonly used for structured data such as employee records,
sales reports, customer information, etc.
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import CSVLoader

from utils.helpers import print_separator, print_title


def main() -> None:
    """Demonstrates CSVLoader."""

    print_title("CSV Loader")

    CSV_FILE_PATH = PROJECT_ROOT / "data" / "input"/ "employees.csv"

    loader = CSVLoader(file_path = CSV_FILE_PATH)

    documents = loader.load()

    print(f"Total Rows Loaded : {len(documents)}")

    print_separator()

    first_row = documents[0]

    print("Row Metadata:\n")
    print(first_row.metadata)

    print_separator()

    print("First Row Content:\n")
    print(first_row.page_content)

if __name__ == "__main__":
    main()