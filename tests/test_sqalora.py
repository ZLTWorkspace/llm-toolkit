from llmtoolkit import print_rank_0
from llmtoolkit.sqalora.config import SQALoraConfig
import torch
import torch.nn as nn

from llmtoolkit.sqalora.layer import Linear

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from llmtoolkit.sqalora.model import SQALoraModel
from llmtoolkit.sqalora.config import SQALoraConfig

BASEMODEL = "/hpc2hdd/home/lzhang330/asset/Llama-2-7b-chat-hf"

def test_sqalora_config():
    config = SQALoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj"],
        init_lora_weights=True,
        use_rslora=False,
        lora_bias=False,
        sparse_preserve_mode=2,
    )
    print_rank_0("SQALoraConfig: ", config)


def test_sqalora_linear(sparse_preserve_mode: int):
    in_features = 8
    out_features = 8
    r = 4
    lora_alpha = 8
    lora_dropout = 0.0
    batch_size = 4

    base_layer = nn.Linear(in_features, out_features)

    sqalora_layer = Linear(
        base_layer=base_layer,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        init_lora_weights=True,
        use_rslora=False,
        lora_bias=False,
        sparse_preserve_mode=sparse_preserve_mode,
    )
    sqalora_layer.to("cuda")
    sqalora_layer.to(torch.bfloat16)
    print_rank_0("BEFORE PRUNE base_layer.weight: ", sqalora_layer.base_layer.weight)

    x = torch.randn(batch_size, in_features).to("cuda").to(torch.bfloat16)

    sqalora_layer.prune(sparsity_ratio=0.25)

    print_rank_0("AFTER PRUNE base_layer.weight: ", sqalora_layer.base_layer.weight)

    sqalora_layer.prune(sparsity_ratio=0.5)

    print_rank_0("mask: ", sqalora_layer.sparse_mask)

    y = sqalora_layer(x)

    print_rank_0("input x: ", x)
    print_rank_0("output y: ", y)

    # validate backward
    # note here all modules are trainable
    y.sum().backward()
    print_rank_0("lora_A.weight: ", sqalora_layer.lora_A.weight)
    print_rank_0("lora_B.weight: ", sqalora_layer.lora_B.weight)
    print_rank_0("lora_A gradient: ", sqalora_layer.lora_A.weight.grad)
    print_rank_0("lora_B gradient: ", sqalora_layer.lora_B.weight.grad)
    print_rank_0("base_layer.weight gradient: ", base_layer.weight.grad)
    print_rank_0(sqalora_layer)


def test_sqalora_model(sparse_preserve_mode: int):
    config = SQALoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.0,
        target_modules=["q_proj"],
        init_lora_weights=True,
        use_rslora=False,
        lora_bias=False,
        sparse_preserve_mode=sparse_preserve_mode,
    )

    pretrained_model_kwargs = {
        "pretrained_model_name_or_path": BASEMODEL,
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(**pretrained_model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_kwargs["pretrained_model_name_or_path"])

    model = SQALoraModel(model=model, config=config)
    model.to("cuda")
    model.prune(sparsity_ratio=0.25)
    print_rank_0(model.calculate_sparsity())
    print_rank_0(model)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

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

    model.save_pretrained(f"sqalora_model_sparse_preserve_mode_{sparse_preserve_mode}")


def test_sqalora_model_load(sparse_preserve_mode: int):
    pretrained_model_kwargs = {
        "pretrained_model_name_or_path": BASEMODEL,
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(**pretrained_model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_kwargs["pretrained_model_name_or_path"])
    model = SQALoraModel.from_pretrained(
        model=model, peft_model_name_or_path=f"sqalora_model_sparse_preserve_mode_{sparse_preserve_mode}"
    )
    model.to("cuda")
    print_rank_0(model)
    print_rank_0(model.calculate_sparsity())
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

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

if __name__ == "__main__":
    test_sqalora_model_load(2)
    exit()
    test_sqalora_config()
    for i in [0,1,2]:
        test_sqalora_linear(i)
        test_sqalora_model(i)
        test_sqalora_model_load(i)
