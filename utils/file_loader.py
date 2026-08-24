import json
from pathlib import Path


def read_text(file_path: str) -> str:
    """Reads a text file."""
    return Path(file_path).read_text(encoding="utf-8")


def write_text(file_path: str, content: str):
    """Writes content to a text file."""
    Path(file_path).write_text(content, encoding="utf-8")


def read_json(file_path: str) -> dict:
    """Reads a JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(file_path: str, data: dict):
    """Writes a dictionary to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)