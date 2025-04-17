#!/bin/bash

TINYLLAMA=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/TinyLlama-1.1B-Chat-v1.0
LLAMA2_7B_CHAT=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Llama-2-7b-chat-hf
LLAMA3_8B_INST=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Meta-Llama-3-8B-Instruct
SDD_OUTPUT=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/tmp

# via VLLM
# mmlu

# single GPU
CUDA_VISIBLE_DEVICES=1 python eval.py --task mmlu --base_model_name_or_path $LLAMA3_8B_INST --backend vllm
# multi GPU
CUDA_VISIBLE_DEVICES=1,2,3,4 python eval.py --task mmlu --base_model_name_or_path $LLAMA3_8B_INST --backend vllm


# via Transformers
# mmlu

# single GPU
CUDA_VISIBLE_DEVICES=1 accelerate launch eval.py --task mmlu --base_model_name_or_path $LLAMA3_8B_INST --backend transformers
# mluti GPU
CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch eval.py --task mmlu --base_model_name_or_path $LLAMA3_8B_INST --backend transformers

