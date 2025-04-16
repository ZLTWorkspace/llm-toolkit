from transformers import (
    AutoTokenizer,
)

from llmtoolkit import (
    DataArguments,
    build_data_module,
)


LLAMA3_8B_INST = "/Users/ltzhang/repo/models/Meta-Llama-3-8B-Instruct"
LLAMA2_7B = "/Users/ltzhang/repo/models/Llama-2-7b-hf"
LLAMA2_7B_CHAT = "/Users/ltzhang/repo/models/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(LLAMA2_7B)
tokenizer.padding_side = "right"
dataargs = DataArguments(
    dataset_name_or_path="mmlu",
    source_max_len=512,
    target_max_len=512,
)
data_module = build_data_module(tokenizer, "mmlu", dataargs)
data_module = build_data_module(tokenizer, "metamath40k", dataargs)
data_module = build_data_module(tokenizer, "alpaca", dataargs)
data_module = build_data_module(tokenizer, "gsm8k", dataargs)
data_module = build_data_module(tokenizer, "codefeedback", dataargs)
