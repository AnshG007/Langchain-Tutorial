"""
File: output_fixing_parser.py

Description
-----------
Demonstrates how OutputFixingParser automatically fixes malformed
LLM outputs before parsing them.

This is useful when the model produces an output that doesn't exactly
match the expected schema.
"""

from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# from langchain.output_parsers import OutputFixingParser
from langchain_classic.output_parsers import OutputFixingParser # type: ignore
# pip install langchain-classic

from llm_client import get_llm
from utils.helpers import print_separator, print_title


class Employee(BaseModel):
    """Schema representing employee information."""

    name: str = Field(description="Employee name")
    department: str = Field(description="Department")
    experience: int = Field(description="Years of experience")
    skills: list[str] = Field(description="Technical skills")


def main() -> None:
    """Demonstrates OutputFixingParser."""

    print_title("Output Fixing Parser")

    llm = get_llm()

    parser = PydanticOutputParser(pydantic_object=Employee)

    fixing_parser = OutputFixingParser.from_llm(
        parser = parser ,
        llm = llm
    )
    
    malformed_output = """
    {
        "name": "Rahul Sharma",
        "department": "AI",
        "experience": "5 years",
        "skills": "Python, SQL, LangChain"
    }
    """

    print("Malformed LLM Output:\n")
    print(malformed_output)

    print_separator()

    parsed_response = fixing_parser.invoke(malformed_output)

    print("Fixed & Parsed Output:\n")
    print(parsed_response)

    print_separator()

    print("Accessing Individual Fields:\n")

    print(f"Name       : {parsed_response.name}")
    print(f"Department : {parsed_response.department}")
    print(f"Experience : {parsed_response.experience}")
    print(f"Skills     : {parsed_response.skills}")


if __name__ == "__main__":
    main()

"""
✅ "experience": "5 years" → 5 (string → integer)
✅ "skills": "Python, SQL, LangChain" → ["Python", "SQL", "LangChain"] (string → list)
"""  