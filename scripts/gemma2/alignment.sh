#!/bin/bash

export PYTHONPATH=$(pwd)

deepspeed griffon/train/train_mem.py \
    --deepspeed scripts/deepspeed/zero2.json \
    --model_name_or_path google/gemma-2-9b-it \
    --version plain \
    --data_path /PATH/TO/llava_pretrain/share-captioner_coco_lcs_sam_1246k_1107.json \
    --image_folder /PATH/TO/ShareGPT4V-images \
    --vision_tower JefferyZhan/clip-vit-large-path14-336_to_1022 \
    --mm_projector_type conv_reduce \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir checkpoints/gemma2/Griffon-Gemma-2-9b-clip1022-projector \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --attn_implementation eager

