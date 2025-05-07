#!/bin/bash

# one-gpu
CUDA_VISIBLE_DEVICES=1 accelerate launch eval.py --task mmlu --base_model_name_or_path /mnt/sdb/zhanglongteng/sdd/zhanglongteng/Meta-Llama-3-8B-Instruct --peft_model_name_or_path /mnt/sdb/zhanglongteng/sdd/zhanglongteng/tmp/validate/Meta-Llama-3-8B-Instruct.mmlu.lr7e-5.sqalora.SR05.SW00.SE03.SS2.SQAT.lorafa/save

# multi-gpu
CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch eval.py --task mmlu --base_model_name_or_path /mnt/sdb/zhanglongteng/sdd/zhanglongteng/Meta-Llama-3-8B-Instruct --peft_model_name_or_path /mnt/sdb/zhanglongteng/sdd/zhanglongteng/tmp/validate/Meta-Llama-3-8B-Instruct.mmlu.lr7e-5.sqalora.SR05.SW00.SE03.SS2.SQAT.lorafa/save
