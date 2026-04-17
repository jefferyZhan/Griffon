#!/bin/bash

# ODinW Inference Script (Thinking Model)
# This script runs inference on the ODinW dataset using vLLM with thinking mode parameters

python run_odinw.py infer \
    --model-path /path/to/Qwen3-VL-Thinking \
    --data-dir /path/to/odinw_data \
    --output-file results/odinw_predictions_thinking.jsonl \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 128000 \
    --max-images-per-prompt 10 \
    --max-new-tokens 32768 \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --repetition-penalty 1.0 \
    --presence-penalty 0.0
