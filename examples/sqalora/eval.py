import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmtoolkit import (
    evaluate_JIT,
    safe_dict2file,
)
from llmtoolkit.sqalora.model import SQALoraModel


def eval_sqalora(pretrained_model_name_or_path, peft_model_name_or_path, task):
    pretrained_model_kwargs = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(**pretrained_model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(peft_model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    model = SQALoraModel.from_pretrained(
        model=model, sqalora_model_name_or_path=peft_model_name_or_path
    )

    model.to("cuda")
    model.eval()
    acc = evaluate_JIT(task, model, tokenizer)

    results = {}
    results["model"] = pretrained_model_name_or_path
    results["peft"] = peft_model_name_or_path
    results["task"] = task
    results["accuracy"] = acc
    safe_dict2file(results, "eval_result.txt")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PEFT models")
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        required=True,
        help="Path to the base model"
    )
    parser.add_argument(
        "--peft_model_name_or_path",
        type=str,
        required=False,
        help="Path to the peft model"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mmlu",
        help="Evaluation task. Default: mmlu"
    )

    args = parser.parse_args()
    eval_sqalora(args.base_model_name_or_path, args.peft_model_name_or_path, args.task)
