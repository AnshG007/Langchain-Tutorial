from llm_client import get_llm
from utils.helpers import print_title, print_separator


def main():

    print_title("LangChain Chatbot Test")
    llm = get_llm()
    print(f"LLM Loaded Successfully: {llm.__class__.__name__}")
    print_separator()
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break
        
        response = llm.invoke(user_input)
        print(f"\nAI: {response.text}\n")

if __name__ == "__main__":
    main()