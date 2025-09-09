import argparse

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmtoolkit import (
    evaluate_JIT,
    safe_dict2file,
)
from llmtoolkit.cclinear import replace_linear_with_cclinear


def eval_cclinear(pretrained_model_name_or_path, task, implementation):
    pretrained_model_kwargs = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda",
    }
    model = AutoModelForCausalLM.from_pretrained(**pretrained_model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    # model.resize_token_embeddings(len(tokenizer))
    model = replace_linear_with_cclinear(model, quant = "fp4", compute_type=torch.bfloat16, residual_dtype=torch.bfloat16, implementation=implementation)


    model.eval()
    acc = evaluate_JIT(task, model, tokenizer)

    results = {}
    results["model"] = pretrained_model_name_or_path
    results["implementation"] = implementation
    results["task"] = task
    results["accuracy"] = acc
    safe_dict2file(results, "eval_result.txt")


if __name__ == "__main__":
    accelerator = Accelerator()

    parser = argparse.ArgumentParser(description="Evaluate PEFT models")
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        required=True,
        help="Path to the base model"
    )
    parser.add_argument(
        "--implementation",
        type=int,
        required=False,
        help="Implementation of residual"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mmlu",
        help="Evaluation task. Default: mmlu"
    )

    args = parser.parse_args()
    eval_cclinear(args.base_model_name_or_path, args.task, args.implementation)
