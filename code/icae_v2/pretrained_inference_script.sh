#!/bin/bash

# BASE_MODEL="mistralai/Mistral-7B-v0.1"
BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
# BASE_MODEL="meta-llama/Llama-2-7b-hf"
# BASE_MODEL="meta-llama/Llama-2-7b-chat-hf"
MODEL_NAME="${MODEL//\//-}"

maxlen=51
mem=128
r=512
mean_compression_rate=4

ICAE_MODEL_PATH=$1 # ICAE model to use; wget "https://huggingface.co/sggetao/icae/resolve/main/mistral_7b_pretrained_icae.safetensors"

python pretrained_inference.py --mean_compression_rate $mean_compression_rate --model_max_length $maxlen --fixed_mem_size $mem --lora_r $r --output_dir $ICAE_MODEL_PATH --model_name_or_path $BASE_MODEL --bf16 --train False
