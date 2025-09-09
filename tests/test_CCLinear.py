import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from llmtoolkit.cclinear import replace_linear_with_cclinear


BASEMODEL = "/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Llama-3.2-1B-Instruct"


def generate(model, tokenizer):
    prompt = "Question: Which of the following types of tests is designed primarily to help predict how successful a person is likely to be in learning new skills?\nA. Achievement\nB. Aptitude\nC. Interest\nD. Personality\n\nAnswer and explanation:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.0,
            temperature=0.1,
        )
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("=" * 40)
    print("Prompt:\n", prompt)
    print("-" * 40)
    print("Response:\n", full_text[len(prompt) :].lstrip())
    print("=" * 40)

def test_CCLinear_prune_0():
    pretrained_model_kwargs = {
        "pretrained_model_name_or_path": BASEMODEL,
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(**pretrained_model_kwargs)
    model.to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_kwargs["pretrained_model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token

    generate(model, tokenizer)

    # model = replace_linear_with_cclinear(model, prune=0.5, compute_type=torch.bfloat16, residual_dtype=torch.bfloat16, implementation=0)
    model = replace_linear_with_cclinear(model, quant = "nf4", compute_type=torch.bfloat16, residual_dtype=torch.bfloat16, implementation=0)

    generate(model, tokenizer)

def test_CCLinear_prune_1():
    pretrained_model_kwargs = {
        "pretrained_model_name_or_path": BASEMODEL,
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(**pretrained_model_kwargs)
    model.to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_kwargs["pretrained_model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token

    generate(model, tokenizer)

    # model = replace_linear_with_cclinear(model, prune=0.5, compute_type=torch.bfloat16, residual_dtype=torch.bfloat16, implementation=1)
    model = replace_linear_with_cclinear(model, quant = "nf4", compute_type=torch.bfloat16, residual_dtype=torch.bfloat16, implementation=1)
    print(model)
    generate(model, tokenizer)

if __name__ == "__main__":
    # test_CCLinear_prune_0()
    test_CCLinear_prune_1()
