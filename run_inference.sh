#!/bin/bash
GPU=$1
MODEL=$2
IMAGE=$3

echo CUDA_VISIBLE_DEVICES=${GPU} python griffon/eval/run_griffon.py \
    --model-path ${MODEL} \
    --image-file ${IMAGE} \
    --query "" \
    --max_new_tokens 3072 \
    --temperature 0 \
    --conv-mode gemma_instruct 

CUDA_VISIBLE_DEVICES=${GPU} python griffon/eval/run_griffon.py \
    --model-path ${MODEL} \
    --image-file ${IMAGE} \
    --query "" \
    --max_new_tokens 3072 \
    --temperature 0 \
    --conv-mode gemma_instruct 