#!/bin/bash

# ODinW Evaluation Script (Instruct Model)
# This script evaluates the inference results using COCO metrics

python run_odinw.py eval \
    --data-dir /PATH/TO/Griffon/Vision-R1/eval/eval_qwen3_vl/odinw \
    --input-file results/odinw_predictions.jsonl \
    --output-file results/odinw_eval_results.json

