#!/bin/bash

# ODinW Inference Script (Instruct Model)
# This script runs inference on the ODinW dataset using vLLM

python run_odinw.py infer \
    --model-path /path/to/Qwen3-VL-Instruct \
    --data-dir /PATH/TO/Griffon/Vision-R1/eval/eval_qwen3_vl/odinw \
    --output-file results/odinw_predictions.jsonl \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 128000 \
    --max-images-per-prompt 10 \
    --max-new-tokens 32768 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --repetition-penalty 1.0 \
    --presence-penalty 1.5

