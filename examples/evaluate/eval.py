import argparse
import shutil

from llmtoolkit import (
    infly_evaluate,
    print_rank_0,
    resize_base_model_and_replace_lmhead_embed_tokens,
    safe_dict2file,
)


def eval_base_model(
    task: str,
    base_model_name_or_path: str,
    load_in_4bit: bool = False,
    backend: str = "vllm",
):
    acc = infly_evaluate(
        task=task,
        model_name_or_path=base_model_name_or_path,
        backend=backend,
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
    load_in_4bit: bool = False,
    backend: str = "vllm",
):
    new_model_name_or_path, new_peft_model_name_or_path = (
        resize_base_model_and_replace_lmhead_embed_tokens(
            base_model_name_or_path=base_model_name_or_path,
            peft_model_name_or_path=peft_model_name_or_path,
        )
    )

    acc = infly_evaluate(
        task=task,
        model_name_or_path=new_model_name_or_path,
        peft_name_or_path=new_peft_model_name_or_path,
        load_in_4bit=load_in_4bit,
        backend=backend,
    )
    results = {}
    results["model"] = base_model_name_or_path
    results["peft"] = peft_model_name_or_path
    results["bits"] = 4 if load_in_4bit else 16
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
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        help="Inference backend to use. Default: vllm"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Whether to load models in 4-bit precision"
    )

    args = parser.parse_args()
    print_rank_0(args)
    if not args.peft_model_name_or_path:
        eval_base_model(
            task=args.task,
            base_model_name_or_path=args.base_model_name_or_path,
            load_in_4bit=args.load_in_4bit,
            backend=args.backend,
        )
    else:
        eval_peft_model(
            task=args.task,
            base_model_name_or_path=args.base_model_name_or_path,
            peft_model_name_or_path=args.peft_model_name_or_path,
            load_in_4bit=args.load_in_4bit,
            backend=args.backend,
        )
