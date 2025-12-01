import os
import json
import warnings
import numpy as np
import re
import torch
import pandas as pd
from datasets import Dataset as HFDataset

REASONING_SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

class CTDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=REASONING_SYSTEM_PROMPT,
        subsample=256,
    ):
        if num_examples > 0:
            warnings.warn("num_examples must be 0 for Countdown dataset. Overriding num_examples to 0.")
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.add_reasoning = add_reasoning
        self.system_prompt = system_prompt
        self.load_test_dataset()

        self.subsample = (
            np.random.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )
        print(f"evaluating {len(self.subsample)} examples")
        assert subsample <= len(self.dataset), "Subsample size is greater than dataset size"
        
    def __len__(self):
        return len(self.subsample)
    
    def load_test_dataset(self):
        self.dataset = []
        cur_path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{cur_path}/../dataset/countdown_cd3_test.jsonl", "r") as f:
            for line in f:
                self.dataset.append(json.loads(line))
        print(len(self.dataset), "examples loaded")

    def __getitem__(self, idx):
        target = int(self.dataset[self.subsample[idx].item()]["output"])
        numbers_str = self.dataset[self.subsample[idx].item()]["input"]
        numbers = [int(num) for num in numbers_str.split(",")]
        question = f"Numbers: {numbers}\nTarget: {target}"
        content = f"{self.system_prompt}\nUsing only the numbers {numbers}, create an arithmetic expression that evaluates to exactly {target}. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>"
        messages = [{"role": "user", "content": content}]
        user_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if self.add_reasoning:
            prompt = user_input + "<reasoning>"
        else:
            prompt = user_input
        return prompt, question, (numbers, target)

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        ).input_ids
        return {"input_ids": input_ids, "questions": questions, "answers": answers, "prompts": prompts}



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


class SudokuDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=SUDOKU_SYSTEM_PROMPT,
        subsample=256,
    ):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.sudoku_file_path = f"{cur_path}/../dataset/4x4_test_sudoku.csv"
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.add_reasoning = add_reasoning
        self.system_prompt = system_prompt
        self.load_test_dataset()

        self.subsample = (
            np.random.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )
        print(f"evaluating {len(self.subsample)} examples")
        assert subsample <= len(self.dataset), "Subsample size is greater than dataset size"
        
    def __len__(self):
        return len(self.subsample)
    
    def load_test_dataset(self):
        """Load the Sudoku dataset from the CSV file."""
        df = pd.read_csv(self.sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
        # Convert pandas DataFrame to HuggingFace Dataset using from_pandas
        self.dataset = HFDataset.from_pandas(df)
        print("Loaded Testing Sudoku dataset with {} examples".format(len(self.dataset)))

    def format_sudoku_grid(self, sudoku_str):
        """Simplified function to format a sudoku string."""
        # Simply pass through the raw string as requested
        return sudoku_str

    def validate_sudoku(self, solution_str, ground_truth=None, question=None):
        if len(question) == 16:
            puzzle_str = question
        else:
            match = re.search(r"Sudoku puzzle: ([0-9]{16})", question)
            if match:
                puzzle_str = match.group(1)
        empty_indices = [i for i in range(16) if puzzle_str[i] == "0"]
        empty_cells = len(empty_indices)
        print(f"Empty cells: {empty_cells}")
        print(puzzle_str)
        if solution_str is None or len(solution_str) == 0:
            return 0, empty_cells, 0.0

        # Handle length issues
        if len(solution_str) < 16:
            # Pad with zeros if too short
            solution_str = solution_str + "0" * (16 - len(solution_str))
        elif len(solution_str) > 16:
            # Truncate if too long
            solution_str = solution_str[:16]

        assert len(puzzle_str) == 16
        # Count correct cells among originally empty cells
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
        accuracy = correct_cells / empty_cells
        return correct_cells, empty_cells, accuracy
    
    def create_prompt(self, input_text):
        # Format similar to your chat function
        if self.num_examples > 0:
            prompt = f"{self.few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n"
        else:
            prompt = input_text
        messages = [{"role": "user", "content": self.system_prompt + "\n\n" + prompt}]
        user_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if self.add_reasoning:
            return user_input + "<reasoning>"
        else:
            return user_input
        
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        puzzle = self.dataset[self.subsample[idx].item()]["Puzzle"]
        solution = self.dataset[self.subsample[idx].item()]["Solution"]

        # Modified question format to reference the examples in the system prompt
        question = f"Solve the following Sudoku puzzle: {puzzle}\n"

        assert len(puzzle) == 16, f"Invalid puzzle length: {len(puzzle)}"

        prompt = self.create_prompt(question)
        return prompt, question, solution

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        ).input_ids
        return {"input_ids": input_ids, "questions": questions, "answers": answers, "prompts": prompts}