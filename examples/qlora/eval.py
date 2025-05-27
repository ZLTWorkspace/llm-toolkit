import argparse
import shutil

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer

from llmtoolkit import (
    evaluate_JIT,
    get_accelerate_model,
    infly_evaluate,
    load,
    print_rank_0,
    resize_base_model_and_replace_lmhead_embed_tokens,
    safe_dict2file,
)
from peft import PeftModel


def quant_and_save(base_model_name_or_path: str, quant_method: str):
    import tempfile

    model, tokenizer = load(base_model_name_or_path, quant_method=quant_method)
    quantized_base_model_path = tempfile.mkdtemp(dir=".")

    model.save_pretrained(quantized_base_model_path)
    tokenizer.save_pretrained(quantized_base_model_path)
    return quantized_base_model_path


def eval_base_model(
    task: str,
    base_model_name_or_path: str,
    load_in_4bit: bool = False,
):
    acc = infly_evaluate(
        task=task,
        model_name_or_path=base_model_name_or_path,
        load_in_4bit=load_in_4bit,
    )

    results = {}
    results["model"] = base_model_name_or_path
    results["bits"] = 4 if load_in_4bit else 16
    results["task"] = task
    results["accuracy"] = acc
    safe_dict2file(results, "eval_result.txt")


def eval_peft_model(
    task: str,
    base_model_name_or_path: str,
    peft_model_name_or_path: str,
    quant_method: str,
):
    # new_model_name_or_path = "/mnt/sdb/zhanglongteng/workspace/llm-toolkit/examples/qlora/tmp2k8r6hou"
    # new_peft_model_name_or_path = "/mnt/sdb/zhanglongteng/workspace/llm-toolkit/examples/qlora/tmpfacizlpp"


    new_model_name_or_path = quant_and_save(base_model_name_or_path, quant_method)
    new_model_name_or_path, new_peft_model_name_or_path = (
        resize_base_model_and_replace_lmhead_embed_tokens(
            base_model_name_or_path=new_model_name_or_path,
            peft_model_name_or_path=peft_model_name_or_path,
        )
    )
    acc = infly_evaluate(
        task=task,
        model_name_or_path=new_model_name_or_path,
        peft_name_or_path=new_peft_model_name_or_path,
        quant_method=quant_method,
    )
    results = {}
    results["model"] = base_model_name_or_path
    results["peft"] = peft_model_name_or_path
    results["quant_method"] = quant_method
    results["task"] = task
    results["accuracy"] = acc
    safe_dict2file(results, "eval_result.txt")

    print_rank_0(
        f"Removing {new_model_name_or_path} and {new_peft_model_name_or_path}."
    )
    try:
        shutil.rmtree(new_model_name_or_path)
        shutil.rmtree(new_peft_model_name_or_path)
    except Exception as e:
        print_rank_0(f"An exception occurred: {e}")

def eval_hqq(base_model_name_or_path, peft_model_name_or_path, task):
    accelerator = Accelerator()
    model, tokenizer = get_accelerate_model(
        base_model_name_or_path,
        True,
        "hqq4",
        None,
        flash_attn=True,
        compute_dtype=torch.bfloat16,
        parallelism="dp",
    )
    tokenizer = AutoTokenizer.from_pretrained(peft_model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_name_or_path)
    model = accelerator.prepare(model)

    model.eval()
    acc = evaluate_JIT(task, model, tokenizer)

    results = {}
    results["model"] = base_model_name_or_path
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
        required=True,
        help="Path to the peft model"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mmlu",
        help="Evaluation task. Default: mmlu"
    )
    parser.add_argument(
        "--quant_method",
        type=str,
        default=None,
        help="Quantization method. Default: None"
    )

    args = parser.parse_args()
    print_rank_0(args)

    eval_hqq(
        task=args.task,
        base_model_name_or_path=args.base_model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
    )
