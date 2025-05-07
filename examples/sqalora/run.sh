#!/bin/bash

LLAMA2_7B=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Llama-2-7b-chat-hf
LLAMA3_8B=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Meta-Llama-3-8B-Instruct


CUDA_VISIBLE_DEVICES=4,5,6,7 DS_SKIP_CUDA_CHECK=1 accelerate launch finetune.py --dataset_name_or_path mmlu --output_dir /mnt/sdb/zhanglongteng/sdd/zhanglongteng/tmp/validate/Meta-Llama-3-8B-Instruct.mmlu.lr7e-5.sqalora.SR05.SW00.SE03.SS2.SQAT.lorafa --logging_strategy steps --logging_steps 1 --save_strategy epoch --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 True --tf32 True --max_steps 100 --hard_padding False --save_total_limit 3 --num_train_epochs 3 --learning_rate 7e-5 --per_device_train_batch_size 1 --source_max_len 896 --target_max_len 128 --model_name_or_path $LLAMA3_8B --flash_attn True --report_to wandb --gradient_checkpointing False --peft sqalora --lora_rank 128 --lora_scale 2.0 --sparse True --sparse_ratio 0.5 --sparse_warmup 0.0 --sparse_end 0.3 --sparse_steps 1 --SQAT True --quant_method nf4 --adamw lorafa
