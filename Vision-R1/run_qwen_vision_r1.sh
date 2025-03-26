#!/bin/bash

export DEBUG_MODE="true"
export LOG_PATH="./checkpoints/Qwen2.5-VL-7B-Instruct-Vision-R1/log.txt"
export WANDB_MODE=offline
export WANDB_HOST=HOST # replace
export PYTHONPATH=$(pwd):$PYTHONPATH

# replace the HOSTFILE, /PATH/TO/Qwen2.5-VL-7B-Instruct
deepspeed --hostfile=HOSTFILE vision_r1/visionr1_qwen.py \
    --deepspeed configs/zero3.json \
    --output_dir checkpoints/Qwen2.5-VL-7B-Instruct-Vision-R1 \
    --model_name_or_path /PATH/TO/Qwen2.5-VL-7B-Instruct \
    --dataset_name JefferyZhan/Vision-R1-Data \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --reward_funcs precision_reward dual_format_reward recall_reward \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-7B-Instruct-Vision-R1 \
    --save_steps 200 \
    --save_only_model true \
    --num_generations 4 \
    --max_completion_length 512 \
    --torch_dtype bfloat16 