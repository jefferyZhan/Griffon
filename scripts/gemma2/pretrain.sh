#!/bin/bash

export PYTHONPATH=$(pwd)

deepspeed griffon/train/train_mem.py \
    --deepspeed ./scripts/deepspeed/zero3.json \
    --model_name_or_path google/gemma-2-9b-it \
    --version gemma_instruct \
    --multi_path scripts/data/pretrain.txt \
    --vision_tower JefferyZhan/clip-vit-large-path14-336_to_1022 \
    --pretrain_mm_mlp_adapter checkpoints/gemma2/Griffon-Gemma-2-9b-clip1022-projector/mm_projector.bin \
    --mm_projector_type conv_reduce \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio square \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/gemma2/Griffon-Gemma-2-9b-clip1022-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --attn_implementation eager
