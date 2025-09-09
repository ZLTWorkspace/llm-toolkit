#!/bin/bash

TINYLLAMA=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/TinyLlama-1.1B-Chat-v1.0
LLAMA2_7B_CHAT=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Llama-2-7b-chat-hf
LLAMA3_8B_INST=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Meta-Llama-3-8B-Instruct
SDD_OUTPUT=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/tmp


CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --task gsm8k --base_model_name_or_path $LLAMA3_8B_INST --implementation 0 &
CUDA_VISIBLE_DEVICES=1 accelerate launch eval.py --task gsm8k --base_model_name_or_path $LLAMA3_8B_INST --implementation 1 &
