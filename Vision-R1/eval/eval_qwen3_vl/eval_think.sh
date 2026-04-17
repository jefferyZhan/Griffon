#!/bin/bash

# ODinW Evaluation Script (Thinking Model)
# This script evaluates the inference results from thinking model using COCO metrics

python run_odinw.py eval \
    --data-dir /path/to/odinw_data \
    --input-file results/odinw_predictions_thinking.jsonl \
    --output-file results/odinw_eval_results_thinking.json

