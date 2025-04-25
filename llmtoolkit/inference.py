import torch
from accelerate import Accelerator
from tqdm import tqdm

from .load_and_save import (
    load,
)
from .utils import (
    ExplicitEnum,
    gsi,
    print_rank_0,
    rank_0,
    require_lib,
)


class InferBackend(ExplicitEnum):
    """
    InferBackend is a class that defines the backend for inference.
    """

    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    SGLANG = "sglang"

    def __str__(self):
        return self.value

    @classmethod
    def choices(cls):
        return [cls.TRANSFORMERS, cls.VLLM, cls.INFLY]


def transformers_inference(
    prompts: list,
    model_name_or_path: str = None,
    peft_name_or_path: str = None,
    max_lora_rank: int = 128,
    max_tokens: int = 1024,
    load_in_4bit: bool = False,
    batch_size: int = 1,
    model = None,
    tokenizer = None,
) -> list:
    if not model or not tokenizer:
        model, tokenizer = load(
            base_model_name_or_path=model_name_or_path,
            peft_model_name_or_path=peft_name_or_path,
            load_in_4bit=load_in_4bit,
        )
    model.eval()
    accelerator = Accelerator()
    tokenizer.padding_side = "left"

    prompts_with_lengths = [(i, prompt, len(tokenizer.tokenize(prompt))) for i, prompt in enumerate(prompts)]
    prompts_with_lengths.sort(key=lambda x: x[2])
    sorted_indices = [x[0] for x in prompts_with_lengths]
    sorted_prompts = [x[1] for x in prompts_with_lengths]

    all_predictions = [None] * len(prompts)

    print(tokenizer.pad_token_id)
    print(tokenizer.eos_token_id)
    print(tokenizer.decode([tokenizer.eos_token_id]))

    num_batches = (len(sorted_prompts) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="Inference Progress"):
        if batch_idx % accelerator.num_processes != accelerator.process_index:
            continue

        start = batch_idx * batch_size
        end = min(start + batch_size, len(sorted_prompts))
        batch_prompts = sorted_prompts[start:end]

        encoded_prompts = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            raw_model = getattr(model, "module", model)
            generated_ids = raw_model.generate(
                **encoded_prompts,
                max_new_tokens=max_tokens,
                top_p=0.0,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_ids = encoded_prompts["input_ids"]
        generated_outputs = []
        for gen_ids, inp_ids in zip(generated_ids, input_ids):
            input_length = len(inp_ids)
            generated_outputs.append(gen_ids[input_length:].tolist())

        batch_predictions = tokenizer.batch_decode(generated_outputs)
        for offset, (input_text, full_output) in enumerate(zip(batch_prompts, batch_predictions)):
            orig_idx = sorted_indices[start + offset]
            all_predictions[orig_idx] = {"prompt": input_text, "response": full_output}

    all_predictions = accelerator.gather_for_metrics(all_predictions)
    if accelerator.is_main_process:
        predictions = [x for x in all_predictions if x is not None]
        return predictions
    else:
        return None


@rank_0
def single_inference(
    model,
    tokenizer,
    input: str,
    task_type: str = "CausalLM",
    source_max_len: str = 512,
    target_max_len: str = 512,
):
    if task_type == "CausalLM":
        inputs = tokenizer(
            input + " ",
            return_tensors="pt",
            max_length=source_max_len,
            truncation=True,
            return_token_type_ids=False,
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=target_max_len,
                eos_token_id=tokenizer.eos_token_id,
                top_p=0.0,
                temperature=0.1,
            )
        pred_text = tokenizer.decode(
            outputs.sequences[0][len(inputs["input_ids"][0]) :],
            skip_special_tokens=True,
        )
    elif task_type == "ConditionalGeneration":
        inputs = tokenizer(input, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=target_max_len)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return pred_text


def vllm_inference(
    prompts: list,
    model_name_or_path: str,
    peft_name_or_path: str = None,
    max_lora_rank: int = 128,
    max_tokens: int = 1024,
    load_in_4bit: bool = False,
) -> list:
    require_lib("vllm")
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    sampling_params = SamplingParams(temperature=0.0, top_p=0.1, max_tokens=max_tokens)

    if gsi.info["n_gpus"] >= 2:
        print_rank_0(
            'WARNING: 2 or more gpus are detected, and VLLM will use all gpus to inference. However, a RuntimeError may raised: "An attempt has been made to start a new process before the current process ...". To avoid this error, wrap your code within " if __name__ == "__main__": ". This is a bug in VLLM, an expected behavior when tp >= 2 & ray. For more info please refer to https://github.com/vllm-project/vllm/pull/5669.'
        )

    if load_in_4bit:
        print_rank_0(
            "For now we only support bitsandbytes quantization for load_in_4bit. This may cause slow inference speed and high GPU memory consumption compared to un-quantized inference. You may consider to decrease the gpu_memory_utilization to avoid OOM. Current gpu_memory_utilization is set to 0.9."
        )
        print_rank_0(
            "WARNING: Please note that no-supprt for bitsandbytes quantization with TP. For more info please refer to https://github.com/vllm-project/vllm/discussions/10117."
        )

    vllm_kwargs = {
        "model": model_name_or_path,
        "dtype": torch.bfloat16,
        "tensor_parallel_size": gsi.info["n_gpus"],
        "gpu_memory_utilization": 0.9,
    }
    if load_in_4bit:
        vllm_kwargs.update({"quantization": "bitsandbytes", "load_format": "bitsandbytes"})
    if peft_name_or_path:
        vllm_kwargs.update({"enable_lora": True, "max_lora_rank": max_lora_rank})

    llm = LLM(**vllm_kwargs)

    generate_kwargs = {}
    if peft_name_or_path:
        generate_kwargs["lora_request"] = LoRARequest(peft_name_or_path, 1, peft_name_or_path)

    outputs = llm.generate(prompts, sampling_params, **generate_kwargs)

    results = [{"prompt": output.prompt, "response": output.outputs[0].text} for output in outputs]

    return results


def batched_inference(
    prompts: list,
    model_name_or_path: str,
    peft_name_or_path: str = None,
    max_lora_rank: int = 128,
    max_tokens: int = 1024,
    load_in_4bit: bool = False,
    backend: InferBackend = InferBackend.VLLM,
) -> list:
    if backend == InferBackend.TRANSFORMERS:
        return transformers_inference(
            prompts,
            model_name_or_path,
            peft_name_or_path=peft_name_or_path,
            max_lora_rank=max_lora_rank,
            max_tokens=max_tokens,
            load_in_4bit=load_in_4bit,
        )
    elif backend == InferBackend.VLLM:
        return vllm_inference(
            prompts,
            model_name_or_path,
            peft_name_or_path=peft_name_or_path,
            max_lora_rank=max_lora_rank,
            max_tokens=max_tokens,
            load_in_4bit=load_in_4bit,
        )
        pass
    elif backend == InferBackend.SGLANG:
        raise NotImplementedError("SGLANG backend is not implemented yet.")
    else:
        raise ValueError(f"Unsupported backend: {backend}. Supported backends are {InferBackend.choices()}.")
