#!/bin/bash


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval.py --task gsm8k --base_model_name_or_path /hpc2hdd/home/lzhang330/asset/Llama-2-7b-chat-hf --peft_model_name_or_path  Llama-2-7b-chat-hf.metamath40k.lorafa.r64.scale2.lr5e-5/save
