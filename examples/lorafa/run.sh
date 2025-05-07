#!/bin/bash


LLAMA2_7B=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Llama-2-7b-chat-hf
LLAMA3_8B=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Meta-Llama-3-8B-Instruct


CUDA_VISIBLE_DEVICES=0,1,2,3 DS_SKIP_CUDA_CHECK=1 accelerate launch finetune.py --dataset_name_or_path metamath40k --output_dir Llama-2-7b-chat-hf.metamath40k.lorafa.r64.scale2.lr5e-5 --logging_strategy steps --logging_steps 1 --save_strategy epoch --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 True --tf32 True --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 3 --learning_rate 5e-5 --per_device_train_batch_size 2 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --report_to wandb --gradient_checkpointing False --peft lora --lora_rank 64 --lora_scale 2.0 --init_lora_weights gaussian --adamw lorafa
