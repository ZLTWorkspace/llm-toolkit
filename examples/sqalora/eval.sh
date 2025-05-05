#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python eval.py --task mmlu --base_model_name_or_path /hpc2hdd/home/lzhang330/asset/Llama-2-7b-chat-hf --peft_model_name_or_path Llama-2-7b-chat-hf.mmlu.sqalora.lr7e-5.SR01.SW01.SE01.SS01/save

CUDA_VISIBLE_DEVICES=0 python eval.py --task mmlu --base_model_name_or_path /hpc2hdd/home/lzhang330/asset/Llama-2-7b-chat-hf --peft_model_name_or_path Llama-2-7b-chat-hf.mmlu.sqalora.lr7e-5.SR01.SW01.SE01.SS01.preserve2/save
