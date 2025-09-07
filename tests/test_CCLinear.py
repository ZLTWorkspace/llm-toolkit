from typing import Union

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from llmtoolkit.sqalora import CCLinear


BASEMODEL = "/Users/ltzhang/repo/models/Llama-3.2-1B-Instruct"



def replace_linear_with_cclinear(
    module: nn.Module,
    quant: str = None,
    prune: Union[float, str] = None,
    compute_type = torch.bfloat16,
    residual_dtype: torch.dtype = torch.float16,
    prefix: str = "",
):

    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if name == "lm_head":
            continue

        if isinstance(child, nn.Linear):
            new_layer = CCLinear(
                child,
                quant=quant,
                prune=prune,
                residual_dtype=residual_dtype,
                compute_type=compute_type,
            )
            setattr(module, name, new_layer)
            print(f"[replace_linear_with_cclinear] Replaced {full_name} ({child.in_features}→{child.out_features})")
        else:
            replace_linear_with_cclinear(
                child,
                quant=quant,
                prune=prune,
                compute_type=compute_type,
                residual_dtype=residual_dtype,
                prefix=full_name,
            )
    return module

def generate(model, tokenizer):
    prompt = "Question: Which of the following types of tests is designed primarily to help predict how successful a person is likely to be in learning new skills?\nA. Achievement\nB. Aptitude\nC. Interest\nD. Personality\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("=" * 40)
    print("Prompt:\n", prompt)
    print("-" * 40)
    print("Response:\n", full_text[len(prompt) :].lstrip())
    print("=" * 40)

def test_CCLinear_prune():
    pretrained_model_kwargs = {
        "pretrained_model_name_or_path": BASEMODEL,
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(**pretrained_model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_kwargs["pretrained_model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token
    model = replace_linear_with_cclinear(model, prune=0.5, compute_type=torch.bfloat16, residual_dtype=torch.bfloat16)
    generate(model, tokenizer)

if __name__ == "__main__":
    test_CCLinear_prune()
