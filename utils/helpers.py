import time


def print_separator(length: int = 60):
    print("=" * length)


def print_title(title: str):
    print_separator()
    print(title)
    print_separator()


def execution_time(start_time: float):
    print(f"\nExecution Time: {time.time() - start_time:.2f} seconds")