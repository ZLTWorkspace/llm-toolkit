#!/bin/bash

#
# This is an example, for lora fine-tuning Llama-2-7b-chat-hf
# Remember to change max_steps to -1 for fine-tuning
#

wandb offline

LLAMA2_7B_CHAT=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Llama-2-7b-chat-hf

#
# Dataset: mmlu
# source_max_len: 896
# target_max_len: 128
#
CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 accelerate launch finetune.py --dataset_name_or_path mmlu --output_dir llama2_7b.mmlu.lora.output --logging_strategy steps --logging_steps 1 --save_strategy epoch --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 True --tf32 True --max_steps 5 --hard_padding False --save_total_limit 3 --num_train_epochs 3 --learning_rate 7e-5 --per_device_train_batch_size 16 --source_max_len 896 --target_max_len 128 --model_name_or_path $LLAMA2_7B_CHAT --flash_attn True --report_to wandb --gradient_checkpointing True --peft lora --lora_rank 16 --lora_scale 2.0

#
# Dataset: metamath40k
# source_max_len: 512
# target_max_len: 512
#
CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 accelerate launch finetune.py --dataset_name_or_path metamath40k --output_dir llama2_7b.metamath40k.lora.output --logging_strategy steps --logging_steps 1 --save_strategy epoch --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 True --tf32 True --max_steps 5 --hard_padding False --save_total_limit 3 --num_train_epochs 3 --learning_rate 3e-4 --per_device_train_batch_size 16 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B_CHAT --flash_attn True --report_to wandb --gradient_checkpointing True --peft lora --lora_rank 16 --lora_scale 2.0

#
# Dataset: wizardlm70k
# source_max_len: 512
# target_max_len: 1024
#
CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 accelerate launch finetune.py --dataset_name_or_path wizardlm70k --output_dir llama2_7b.wizardlm70k.lora.output --logging_strategy steps --logging_steps 1 --save_strategy epoch --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 True --tf32 True --max_steps 5 --hard_padding False --save_total_limit 3 --num_train_epochs 3 --learning_rate 3e-4 --per_device_train_batch_size 16 --source_max_len 512 --target_max_len 1024 --model_name_or_path $LLAMA2_7B_CHAT --flash_attn True --report_to wandb --gradient_checkpointing True --peft lora --lora_rank 16 --lora_scale 2.0

#
# Dataset: codefeedback
# source_max_len: 512
# target_max_len: 1024
#
CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 accelerate launch finetune.py --dataset_name_or_path codefeedback --output_dir llama2_7b.codefeedback.lora.output --logging_strategy steps --logging_steps 1 --save_strategy epoch --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 True --tf32 True --max_steps 5 --hard_padding False --save_total_limit 3 --num_train_epochs 3 --learning_rate 3e-4 --per_device_train_batch_size 16 --source_max_len 512 --target_max_len 1024 --model_name_or_path $LLAMA2_7B_CHAT --flash_attn True --report_to wandb --gradient_checkpointing True --peft lora --lora_rank 16 --lora_scale 2.0