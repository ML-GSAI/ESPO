# Adapted based on huggingface/open-r1: https://github.com/huggingface/open-r1/blob/6a0cd5c8ad031fc75118a4ce7f42a4860c3d8dea/src/open_r1/rewards.py


"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict, Optional
import numpy as np
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils.code_providers import get_provider

from .utils.local_sandbox import local_execute




def extract_code(completion: str, language: str = "python") -> str:
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[0] if len(matches) >= 1 else ""
    return extracted_answer


def code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """Reward function that evaluates code snippets using a code execution provider.

    Assumes the dataset contains a `verification_info` column with test cases.

    Args:
        completions: List of model completions to evaluate
        num_parallel: Number of parallel code executions (default: 2)
        provider_type: Which code execution provider to use (default: "e2b")
        enforce_same_language: If True, verify all problems use the same language (default: False)
        **kwargs: Additional arguments passed to the verification
    """
    evaluation_script_template = """
import subprocess
import json

def evaluate_code(code, test_cases):
    passed = 0
    total = len(test_cases)
    exec_timeout = 5

    for case in test_cases:
        process = subprocess.run(
            ["python3", "-c", f"{{code}}\\n{{case}}"],
            text=True,
            capture_output=True,
            timeout=exec_timeout
        )

        if process.returncode != 0:  # Error in execution
            continue

        # If we get here, the assertion passed (no error)
        passed += 1

    success_rate = (passed / total)
    return success_rate

code_snippet = {code_literal}
test_cases = {test_cases_literal}
rate = evaluate_code(code_snippet, test_cases)
print("__PASS_RATE__", rate)
"""
    # 1. compute format rewards
    format_rewards = get_code_format_reward(language='python')(completions)

    # 2. collect scripts and their indices in the original array
    template = evaluation_script_template
    scripts = []
    valid_indices = []
    pairs = []
    for i, (reward, completion) in enumerate(zip(format_rewards, completions)):
        if reward < 1:
            continue
        code = extract_code(completion[-1]["content"])
        tc = kwargs["test_cases"][i]
        scripts.append(
            template.format(
                code_literal=repr(code),
                test_cases_literal=repr(tc),
            )
        )
        valid_indices.append(i)
        pairs.append((code, tc))

    if provider_type in [ "e2b","morph" ]:
    # 3. execute scripts in parallel
        execution_provider = get_provider(
            provider_type=provider_type,
            num_parallel=num_parallel,
            **kwargs,
        )
        results = execution_provider.execute_scripts(scripts, ["python"] * len(scripts))
    else:
        assert provider_type == "local", "Invalid provider type"
        results = local_execute(pairs)


    # 4. fill results into a list of the same length as completions, and keep None for reward=0
    final_results = [0.0] * len(completions)
    for idx, res in zip(valid_indices, results):
        final_results[idx] = res

    return final_results



def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    import ast
    pattern = re.compile(
        rf"^"
        r"(?:(?!```)[\s\S])*?"
        rf"```{language}\n"    # match ```language\n
        r"(?:(?!```)[\s\S])*?"         # match any character, but not ```
        rf"```\n?$",                     # match ``` and end of string
        re.DOTALL
    )

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for content in completion_contents:
            # # First check if the format matches
            # import pdb; pdb.set_trace();
            format_match = pattern.fullmatch(content)
            if not format_match:
                rewards.append(0.0)
                continue
                
            # Extract code from between code blocks
            code_blocks = re.findall(rf"```{language}\n(.*?)```", content, re.DOTALL)
            if not code_blocks:
                rewards.append(0.0)
                continue
                
            # Get the first code block (in case there are multiple)
            code = code_blocks[0].strip()
            
            # Check syntax if it's Python code
            if language == "python":
                try:
                    ast.parse(code)
                    syntax_valid = True
                except SyntaxError:
                    syntax_valid = False
                except: 
                    syntax_valid = False
                    print('Other error')
                rewards.append(1.0 if syntax_valid else 0.5) #  grammar error get partial reward
            else:
                # For other languages, just check format for now
                rewards.append(1.0)
                
        return rewards

    return code_format_reward




def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def extract_answer(text: str) -> str:
    """
    General answer extraction logic:
    - Prefer matching "The answer is X" (gsm8k_cot style)
    - Then match "#### X" (gsm8k official style)
    - If neither, fallback to the last number
    """
    # 1. CoT 格式：The answer is 42
    match_cot = re.search(r"The answer is (\-?[0-9\.,]+).", text)
    if match_cot:
        return match_cot.group(1).strip()
    
    # 2. GSM8K 格式：#### 42
    match_hash = re.search(r"#### (\-?[0-9\.,]+)", text)
    if match_hash:
        return match_hash.group(1).strip()
    
    # 3. fallback：提取最后一个数字
    fallback_match = re.findall(r"(\-?[0-9]+)", text)
    if fallback_match:
        return fallback_match[-1].strip()
    
    return ""  # 没找到数字

def correctness_reward_func_gsm8k(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    """
    Calculate accuracy reward:
    - Return 1.0 for exact match, else 0.0
    """

    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_answer(r) for r in responses]

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Prompt:{RESET}\n{q}\n",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}\n",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}\n",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}\n",
    )

    return [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    return matches[-1].strip() if matches else None


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except:
        return False


def evaluate_equation(equation_str):
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})
    except:
        return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str)
    do_print = np.random.rand() < 0.4

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score

    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


def reward_func_countdown(prompts, completions, run_name, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        ground_truth = {"target": kwargs["target"][i], "numbers": kwargs["numbers"][i]}
        scores.append(compute_score(response, ground_truth))

    return scores


def extract_answer_sudoku(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        return "".join(char for char in matches[-1].strip() if char.isdigit())
    return None


def validate_sudoku_solution(solution_str, ground_truth, puzzle):
    if solution_str is None or len(solution_str) == 0:
        return 0.0

    if len(solution_str) < 16:
        # Pad with zeros if too short
        solution_str = solution_str + "0" * (16 - len(solution_str))
    elif len(solution_str) > 16:
        # Truncate if too long
        solution_str = solution_str[:16]

    empty_indices = [i for i in range(16) if puzzle[i] == "0"]

    if empty_indices:
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
        return correct_cells / len(empty_indices)
    return 0.0


def reward_func_sudoku(prompts, completions, run_name, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        do_print = np.random.rand() < 0.4
        puzzle = kwargs["puzzle"][i]
        ground_truth = kwargs["solution"][i]
        solution = extract_answer_sudoku(response)

        score = 0.0 if solution is None else validate_sudoku_solution(solution, ground_truth, puzzle)
        scores.append(score)

        if do_print:
            print(f"--------------------------------")
            print(f"Puzzle: {puzzle} (length: {len(puzzle)})")
            print(f"Extracted solution: {solution}  (length: {len(solution) if solution else 0})")
            print(f"Ground_truth: {ground_truth}")
            print(f"Score: {score:.4f}")

    return scores



def correctness_reward_func_math(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    parsed_responses = []
    for r in responses:
        parsed_responses.append(parse(r))
    parsed_answer = [parse(a) for a in answer]
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Question:{RESET}\n{q}",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{parsed_answer[0]}",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{parsed_responses[0]}",
    )
    print("✅" if verify(parsed_responses[0], parsed_answer[0]) else "❌")

    return [1.0 if verify(r, a) else 0.0 for r, a in zip(parsed_responses, parsed_answer)]


def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "gsm8k": [correctness_reward_func_gsm8k],
        "math": [correctness_reward_func_math],
        "countdown": [reward_func_countdown],
        "sudoku": [reward_func_sudoku],
        "code": [get_code_format_reward(language=script_args.code_language),
                 update_wrapper(
                    partial(
                        code_reward,
                        num_parallel=script_args.parallel_code_exec_per_proc,
                        provider_type=script_args.code_provider,
                        enforce_same_language=getattr(script_args, "enforce_same_language", False),
                    ),
                    code_reward,
                )],
    }
    reward_funcs = REWARD_FUNCS_REGISTRY.get(script_args.dataset_name, [])
    assert reward_funcs, f"No reward functions found for dataset {script_args.dataset_name}"

    return reward_funcs
