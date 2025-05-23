import os
import re
from abc import ABC, abstractmethod

import lm_eval
from tqdm import tqdm
from transformers import AutoTokenizer

from .dataset import (
    build_data_module,
)
from .inference import (
    InferBackend,
    batched_inference,
    transformers_inference,
)
from .utils import (
    create_timestamp,
    get_rank,
    gsi,
    print_rank_0,
    require_lib,
    safe_dict2file,
    safe_list2file,
)


"""
Note:
1. The number of fewshot of truthfulqa_mc1 is set to 0, however, TruthfulQA is technically a 6-shot task in the Harness because each example is prepended with 6 Q/A pairs, even in the 0-shot setting.
2. To eval the baseline score of gsm8k, we are better to finetune the model on the full GSM8K training set for 50 epochs.
"""
task2shot = {
    "mmlu": 5,
    "gsm8k": 5,
    "winogrande": 5,
    "piqa": 5,
    "hellaswag": 10,
    "truthfulqa_mc1": 0,
    "arc_challenge": 25,
    "openbookqa": 5,
}


def vllm_lm_eval(
    task: str,
    model_name_or_path: str,
    shot: int = None,
    dump: bool = False,
    output_dir: str = None,
) -> list:
    require_lib("vllm")
    task_manager = lm_eval.tasks.TaskManager()
    if shot is None:
        if task in task2shot.keys():
            shot = task2shot[task]
        else:
            shot = 0

    model_args = f"pretrained={model_name_or_path},tensor_parallel_size={gsi.n_gpus},dtype=bfloat16,max_model_len=4096"

    print_rank_0(f"evaluating {task}")
    result = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=[task],
        num_fewshot=shot,
        task_manager=task_manager,
        batch_size="auto",
    )

    if dump:
        if get_rank() == 0:
            if output_dir is None:
                output_dir = f"eval_{create_timestamp()}"
            safe_dict2file(
                {"model": model_name_or_path},
                os.path.join(output_dir, "eval_result.txt"),
            )
            safe_dict2file(result, os.path.join(output_dir, "eval_result.txt"))

    return result


def hf_lm_eval(
    task: str,
    model_name_or_path: str,
    peft_name_or_path: str = None,
    shot: int = None,
    dump: bool = False,
    output_dir: str = None,
) -> list:
    task_manager = lm_eval.tasks.TaskManager()
    if shot is None:
        if task in task2shot.keys():
            shot = task2shot[task]
        else:
            shot = 0

    model_args = (
        f"pretrained={model_name_or_path},tokenizer={model_name_or_path},dtype=bfloat16"
    )
    if peft_name_or_path is not None:
        model_args += f",peft={peft_name_or_path}"

    print_rank_0(f"evaluating {task}")
    result = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=[task],
        num_fewshot=shot,
        task_manager=task_manager,
        batch_size="auto",
    )

    if dump:
        if get_rank() == 0:
            if output_dir is not None:
                output_dir = f"eval_{create_timestamp()}"
            safe_dict2file(
                {"model": model_name_or_path},
                os.path.join(output_dir, "eval_result.txt"),
            )
            safe_dict2file(result, os.path.join(output_dir, "eval_result.txt"))

    return result


"""
We also provide some other custom eval fuctions.

The evaluate data should be the format below.
data = [
    {"prompt": "Q1", "golden": "This is the correct answer", "predicate": "This is the predicate answer"},
    {"prompt": "Q2", "golden": "This is the correct answer", "predicate": "This is the predicate answer"},
]

usage:
> task = 'gsm8k'
> accuracy = evaluate(data, task)
> print(f"Accuracy: {accuracy:.2f}")
"""


class EvaluationStrategy(ABC):
    @abstractmethod
    def is_correct(self, golden: str, predicate: str) -> bool:
        pass


class DefaultEvaluationStrategy(EvaluationStrategy):
    def is_correct(self, golden: str, predicate: str) -> bool:
        return golden == predicate


class GSM8KEvaluationStrategy(EvaluationStrategy):
    def is_correct(self, golden: str, predicate: str) -> bool:
        golden_numbers = self.extract_numbers(golden)
        predicate_numbers = self.extract_numbers(predicate)
        return golden_numbers == predicate_numbers

    def extract_numbers(self, text: str):
        regex = r"(-?(0|([1-9][0-9]*))(\.[\d]+)?)"
        matches = re.findall(regex, text)
        if matches:
            result = matches[-1][0]
        else:
            result = 0

        try:
            return int(result)
        except ValueError:
            try:
                return float(result)
            except ValueError:
                print(f"'{result}' is invalid thus cannot be transformed into numbers")
                return 0


class MMLUEvaluationStrategy(EvaluationStrategy):
    def is_correct(self, golden: str, predicate: str) -> bool:
        golden_choice = golden.strip()
        predicate_choice = predicate.strip()
        if len(predicate_choice) != 0 and len(golden_choice) != 0:
            golden_choice = golden_choice[0].upper()
            predicate_choice = predicate_choice[0].upper()
            if golden_choice not in ["A", "B", "C", "D"]:
                raise ValueError(
                    f"The first letter of label '{golden}' not in A, B, C, D. Aborting."
                )
            return golden_choice == predicate_choice
        else:
            return False


class Evaluator:
    def __init__(self, strategy: EvaluationStrategy):
        self.strategy = strategy

    def evaluate(self, data: list) -> float:
        total = len(data)
        correct = sum(
            self.strategy.is_correct(item["golden"], item["predicate"])
            for item in tqdm(data, desc="Evaluating")
        )
        return correct / total if total > 0 else 0.0


def offline_evaluate(task: str, data: list) -> float:
    if task == "gsm8k":
        strategy = GSM8KEvaluationStrategy()
    else:
        strategy = DefaultEvaluationStrategy()

    evaluator = Evaluator(strategy)
    return evaluator.evaluate(data)


def evaluate_JIT(
    task: str,
    model,
    tokenizer,
) -> float:
    if task == "gsm8k":
        strategy = GSM8KEvaluationStrategy()
    elif task == "mmlu":
        strategy = MMLUEvaluationStrategy()
    else:
        strategy = DefaultEvaluationStrategy()

    evaluator = Evaluator(strategy)
    eval_dataset = build_data_module(tokenizer, task)["eval_dataset"]
    prompts = list(eval_dataset["input"])
    prompt_to_golden = {item["input"]: item["output"] for item in eval_dataset}

    # since the evaluation is based on the keywords in the output
    # thus we only keep sufficient sequence
    max_tokens = {
        "mmlu": 32,
        "gsm8k": 1024,
    }

    results = transformers_inference(prompts=prompts, max_tokens=max_tokens[task], model=model, tokenizer=tokenizer)

    if get_rank() != 0 or results is None:
        return None

    inspection = []
    for result in results:
        prompt = result["prompt"]
        response = result["response"]
        golden = prompt_to_golden.get(prompt, None)
        if golden is not None:
            inspection.append(
                {"prompt": prompt, "golden": golden, "predicate": response}
            )
    safe_list2file(inspection, f"eval_{create_timestamp()}.json")
    return evaluator.evaluate(inspection)


def infly_evaluate(
    task: str,
    model_name_or_path,
    peft_name_or_path: str = None,
    quant_method: str = None,
    backend: InferBackend = InferBackend.VLLM,
) -> float:
    if task == "gsm8k":
        strategy = GSM8KEvaluationStrategy()
    elif task == "mmlu":
        strategy = MMLUEvaluationStrategy()
    else:
        strategy = DefaultEvaluationStrategy()

    evaluator = Evaluator(strategy)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    eval_dataset = build_data_module(tokenizer, task)["eval_dataset"]
    prompts = list(eval_dataset["input"])
    prompt_to_golden = {item["input"]: item["output"] for item in eval_dataset}

    # since the evaluation is based on the keywords in the output
    # thus we only keep sufficient sequence
    max_tokens = {
        "mmlu": 32,
        "gsm8k": 1024,
    }

    results = batched_inference(
        prompts, model_name_or_path, peft_name_or_path, quant_method=quant_method, max_tokens=max_tokens.get(task, 1024), backend=backend
    )

    if get_rank() != 0 or results is None:
        return None

    inspection = []
    for result in results:
        prompt = result["prompt"]
        response = result["response"]
        golden = prompt_to_golden.get(prompt, None)
        if golden is not None:
            inspection.append(
                {"prompt": prompt, "golden": golden, "predicate": response}
            )
    safe_list2file(inspection, f"eval_{create_timestamp()}.json")
    return evaluator.evaluate(inspection)
