import json
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

load_dotenv()

def load_config(config_path: str = 'config.json') -> dict:
    """Loads a JSON configuration file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)

def get_llm():
    config = load_config() 
    provider = config["provider"].lower()
    #print("GEMINI_API_KEY:", os.getenv("OPENAI_API_KEY"))
    if provider == "openai":
        return ChatOpenAI(
            model = config["openai"]['model'],
            temperature = config["openai"]['temperature'],
            max_tokens = config["openai"]['max_tokens'],
            api_key = os.getenv("OPENAI_API_KEY")
        )

    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model = config["gemini"]['model'],
            temperature = config["gemini"]['temperature'],
            max_output_tokens = config["gemini"]['max_tokens'],
            api_key = os.getenv("GEMINI_API_KEY")
        )

    elif provider == "anthropic":
        return ChatAnthropic(
            model = config["anthropic"]['model'],
            temperature = config["anthropic"]['temperature'],
            max_tokens = config["anthropic"]['max_tokens'],
            api_key = os.getenv("ANTHROPIC_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")