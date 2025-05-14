import os

import torch
import transformers
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HqqConfig,
)

from peft import (
    PeftModel,
)

from .model import (
    auto_add_special_tokens,
)
from .sparse import (
    apply_sparse,
)
from .utils import (
    create_timestamp,
    is_ipex_available,
    print_rank_0,
    rank_0,
)


def load(
    base_model_name_or_path: str,
    peft_model_name_or_path: str = None,
    quant_method: str = None,
):
    compute_dtype = torch.bfloat16

    # TODO check flash-attn
    model_kwargs = {"pretrained_model_name_or_path": base_model_name_or_path}
    if torch.cuda.is_available():
        model_kwargs.update(
            {"torch_dtype": compute_dtype, "attn_implementation": "flash_attention_2", "device_map": "cuda"}
        )
    if quant_method == "nf4":
        model_kwargs.update(
            {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_quant_storage="uint8",
                )
            }
        )
    elif quant_method == "fp4":
        model_kwargs.update(
            {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="fp4",
                    bnb_4bit_quant_storage="uint8",
                )
            }
        )
        pass
    elif quant_method == "hqq4":
        quant_config = HqqConfig(nbits=4, group_size=64)
        model_kwargs.update({"quantization_config": quant_config})
    else:
        raise ValueError(f"Unsupported quantization method {quant_method}")
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    target_tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

    # make sure the special tokens are the same as the base model when training
    auto_add_special_tokens(model, target_tokenizer)

    if peft_model_name_or_path:
        peft_tokenizer = AutoTokenizer.from_pretrained(peft_model_name_or_path)
        if len(target_tokenizer) != len(peft_tokenizer):
            print(
                f"Since the embedding of base model mismatch peft adapter ({len(target_tokenizer)} - {len(peft_tokenizer)}), resizing."
            )
            model.resize_token_embeddings(len(peft_tokenizer))
        target_tokenizer = peft_tokenizer
        model = PeftModel.from_pretrained(model, peft_model_name_or_path)

    return model, target_tokenizer


def resize_base_model_and_replace_lmhead_embed_tokens(
    base_model_name_or_path: str,
    peft_model_name_or_path: str,
):
    """
    TODO: Copy all the files from old dir to new dir.
    """
    import json
    import tempfile

    from safetensors.torch import load_file, save_file

    if torch.cuda.is_available():
        device_map = "cuda"
    else:
        device_map = None
    # 1. Load the base model, adapter model, and adapter config
    model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=torch.bfloat16, device_map=device_map)
    adapter_model = load_file(f"{peft_model_name_or_path}/adapter_model.safetensors")
    with open(f"{peft_model_name_or_path}/adapter_config.json", encoding="utf-8") as file:
        adapter_config = json.load(file)

    # 2. Check if the base model need to be resized
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    peft_tokenizer = AutoTokenizer.from_pretrained(peft_model_name_or_path)
    if len(base_tokenizer) != len(peft_tokenizer):
        print_rank_0(
            f"Resizing the embedding of base model, to match the length of peft adapter, from {len(base_tokenizer)} to {len(peft_tokenizer)}."
        )
        model.resize_token_embeddings(len(peft_tokenizer))

    # 3. Replace the lm_head and embed_tokens (i.e., unsupport lora wight in vLLM) with the adapter model
    # TODO: Check if lm_head and embed_tokens are in the adapter model
    support_lora_names = [
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "bias",
    ]
    support_adapter_model = {}
    unsupport_adapter_model = {}

    for key, value in adapter_model.items():
        if any(substring in key for substring in support_lora_names):
            support_adapter_model[key] = value
        else:
            unsupport_adapter_model[key] = value

    for key, value in unsupport_adapter_model.items():
        for n, m in model.named_parameters():
            if key in n:
                print_rank_0(f"Replace {n} with {key}.")
                m.data = value
                break

    # 4. Save the resized model and adapter model
    resized_base_model_path = tempfile.mkdtemp(dir=".")
    resized_adapter_model_path = tempfile.mkdtemp(dir=".")

    # Save the resized model and tokenizer
    # Note that the tokenizer is from the adapter model, not the base model
    model.save_pretrained(resized_base_model_path)
    peft_tokenizer.save_pretrained(resized_base_model_path)

    # Save the resized adapter model and tokenizer
    save_file(support_adapter_model, f"{resized_adapter_model_path}/adapter_model.safetensors")
    with open(f"{resized_adapter_model_path}/adapter_config.json", "w", encoding="utf-8") as file:
        json.dump(adapter_config, file, ensure_ascii=False, indent=4)
    peft_tokenizer.save_pretrained(resized_adapter_model_path)

    print_rank_0(
        f"Finish replacing. The new base model is saved at {resized_base_model_path}. The new adapter model is saved at {resized_adapter_model_path}."
    )

    # 5. Return the path of the resized model and adapter model
    return resized_base_model_path, resized_adapter_model_path


def flexible_load(args):
    if args.flash_attn:
        import importlib.util

        flashattn_spec = importlib.util.find_spec("flash-attn")
        if flashattn_spec is None:
            raise FileNotFoundError("You can not use flash_attn now since flash-attn was not installed.")

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()

    max_memory = f"{args.max_memory_MB}MB"
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = None

    if args.device_map is not None:
        # if we are in a distributed setting, we need to set the device map and max memory per device
        if os.environ.get("LOCAL_RANK") is not None:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device_map = {"": local_rank}
            max_memory = {"": max_memory[local_rank]}

    if args.deepspeed is not None:
        print_rank_0("Using deepspeed, disabling device_map...")
        device_map = None

    if not args.quant:
        assert args.bits in [16, 32]

    print_rank_0(f"loading base model {args.model_name_or_path}...")
    compute_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)

    if args.quant:
        print_rank_0("LOADING QUANTIZED MODEL")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map=device_map,
            max_memory=max_memory,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type,
            ),
            torch_dtype=(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token,
            attn_implementation="flash_attention_2" if args.flash_attn else "eager",
        )
    else:
        print_rank_0("LOADING UNQUANTIZED MODEL")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map=device_map,
            torch_dtype=(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token,
            attn_implementation="flash_attention_2" if args.flash_attn else "eager",
        )
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print_rank_0("=" * 80)
            print_rank_0("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print_rank_0("=" * 80)

    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        print_rank_0("Intel XPU does not support float16 yet, so switching to bfloat16")

    # setattr(model, 'model_parallel', True)
    # setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "right"

    # add special tokens
    # 1. add pad_token if pad_token is None, as unk_token or eos_token if unk_token is None
    # 2. add unk_token if unk_token is None, as pad_token or eos_token if pad_token is None
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = (
            tokenizer.unk_token
            if tokenizer.unk_token is not None
            else tokenizer.convert_ids_to_tokens(model.config.eos_token_id)
        )
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = tokenizer.convert_ids_to_tokens(model.config.eos_token_id)
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = tokenizer.convert_ids_to_tokens(model.config.bos_token_id)
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = (
            tokenizer.pad_token
            if tokenizer.pad_token is not None
            else tokenizer.convert_ids_to_tokens(model.config.eos_token_id)
        )

    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
    print_rank_0(f"pad_token: {tokenizer.pad_token}")
    print_rank_0(f"eos_token: {tokenizer.eos_token}")
    print_rank_0(f"bos_token: {tokenizer.bos_token}")
    print_rank_0(f"unk_token: {tokenizer.unk_token}")

    if args.peft_name_or_path:
        print_rank_0("Loading adapter")
        model = PeftModel.from_pretrained(model, args.peft_name_or_path)
        if args.unify_load:
            model = model.merge_and_unload()

    return model, tokenizer


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    print_rank_0(f"adding special tokens, {special_tokens_dict}")
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


@rank_0
def merge_and_save(model_name_or_path, peft_path, output_path):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_path)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    save_url = os.path.join(output_path, f"merged_model_{create_timestamp()}")
    model.save_pretrained(save_url)
    tokenizer.save_pretrained(save_url)
    print_rank_0(f"Merged model has been successfully saved at {save_url}.")
    del model, tokenizer
