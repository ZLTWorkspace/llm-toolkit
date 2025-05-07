#!/bin/bash


CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --task gsm8k --base_model_name_or_path /hpc2hdd/home/lzhang330/asset/Llama-2-7b-chat-hf --peft_model_name_or_path  Llama-2-7b-chat-hf.metamath40k.qlora.r64.scale2.lr5e-5/save --load_in_4bit True
