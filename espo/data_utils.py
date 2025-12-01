from datasets import load_dataset, Dataset
import pandas as pd
import random
import numpy as np
import torch
import os
from espo.rewards import extract_hash_answer

# Constants for prompts
REASONING_SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

**Rules:**
- Fill empty cells with digits 1-4.
- Each row must contain digits 1-4 exactly once.
- Each column must contain digits 1-4 exactly once.
- Each 2x2 box must contain digits 1-4 exactly once.

**Example:**
Puzzle: 0401002010030310
This puzzle grid looks like this:
0 4 | 0 1
0 0 | 2 0
----+----
1 0 | 0 3
0 3 | 1 0

Solution: 2431312412434312
The solved grid looks like this:
2 4 | 3 1
3 1 | 2 4
----+----
1 2 | 4 3
4 3 | 1 2

**Important:** Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""

# Format into conversation
def make_conversation(example):
    prompt = []

    prompt.append({"role": "system", "content":"""You are a helpful assistant. When you output code, it must be in a single fenced code block using triple backticks (```) and the language specifier (e.g., ```python). The code block must appear as the last part of your response and end with a newline after the closing backticks (```). Do not include any text after the code block. Do not include extra code blocks or explanations outside the final code block."""})

    prompt.append({"role": "user", "content": example["question"] + ('\nTest cases: ' + example['test_cases'][0] if '\n```\n' not in example["question"] else '')})
    return {"prompt": prompt}
    

def get_code_questions(split="train") -> Dataset:
    """Load code questions from a JSON file and return as a Dataset."""

    file_path = "./recipes/acecode_hard.jsonl"
    dataset = load_dataset("json", data_files=file_path)
    dataset = dataset.map(make_conversation)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    return dataset[split]

def get_gsm8k_questions(split="train"):
    data = load_dataset("openai/gsm8k", "main")[split]
    
    def format_example(x):
        return {
            "prompt": [
                {"role": "user", "content": f"{x['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."},
            ],
            "answer": extract_hash_answer(x["answer"])
        }
    
    return data.map(format_example)


def get_math_questions(split="train") -> Dataset:
    data = load_dataset("ankner/math-500", split=split)  # type: ignore
    data = data.map(
        lambda x: {  
            "prompt": [
                {"role": "user", "content": f"{x['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."},
            ],
            "answer": x["solution"],
        }
    )  
    return data  

def get_countdown_questions(split="train") -> Dataset:
    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)
    data = data.filter(lambda x: len(x["nums"]) == 3)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{REASONING_SYSTEM_PROMPT}\nUsing only the numbers {x['nums']}, create an arithmetic expression that evaluates to exactly {x['target']}. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>",
                },
            ],
            "target": x["target"],
            "numbers": x["nums"],
        }
    )


def get_sudoku_questions() -> Dataset:
    """Load the Sudoku dataset for training or evaluation."""
    cur_path = os.path.dirname(os.path.abspath(__file__))
    sudoku_file_path = "../dataset/4x4_sudoku_unique_puzzles.csv"
    
    # Load and process original data
    sudoku_file_path = os.path.join(cur_path, sudoku_file_path)
    df = pd.read_csv(sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)

    processed_data = data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SUDOKU_SYSTEM_PROMPT}\n\nNow, solve the following Sudoku puzzle: {x['Puzzle']}\n",
                },
            ],
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
        }
    )
    
    # Save to cache
    # processed_data.save_to_disk(cache_path)
    return processed_data


def get_datasets(name:str) -> Dataset:
    if name == "code":
        return get_code_questions()
    elif name == "gsm8k":
        return get_gsm8k_questions()
    elif name == "math":
        return get_math_questions()
    elif name == "countdown":
        return get_countdown_questions()
    elif name == "sudoku":
        return get_sudoku_questions()
    else:
        raise ValueError(f"Dataset {name} not supported.")