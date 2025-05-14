#!/bin/bash

LLAMA2_7B=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Llama-2-7b-chat-hf
LLAMA3_8B=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Meta-Llama-3-8B-Instruct

CUDA_VISIBLE_DEVICES=7 accelerate launch eval.py --task gsm8k --base_model_name_or_path $LLAMA2_7B --peft_model_name_or_path  /mnt/sdb/zhanglongteng/sdd/zhanglongteng/finetune/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct.mmlu.hqq4.r16.scale2.lr5e-5/checkpoint-9360 --quant_method hqq4
