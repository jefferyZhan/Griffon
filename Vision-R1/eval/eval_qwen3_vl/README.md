# ODinW Benchmark Evaluation

This directory contains the implementation for evaluating vision-language models on the ODinW (Object Detection in the Wild) 13 dataset using vLLM for high-speed inference.

## Overview

ODinW is a comprehensive object detection benchmark that consists of 13 diverse datasets spanning various domains. This implementation provides:

- **High-speed inference** using vLLM with automatic batch optimization
- **Unified evaluation** across 13 diverse object detection datasets
- **COCO-style metrics** including mAP, mAP_50, mAP_75, etc.
- **Modular code structure** for easy maintenance and extension

## Project Structure

```
ODinW-13/
├── run_odinw.py          # Main script for inference and evaluation
├── dataset_utils.py      # Dataset loading and preprocessing utilities
├── eval_utils.py         # Evaluation logic and COCO metrics computation
├── infer_instruct.sh     # Inference script for instruct models
├── infer_think.sh        # Inference script for thinking models
├── eval_instruct.sh      # Evaluation script for instruct model results
├── eval_think.sh         # Evaluation script for thinking model results
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Requirements

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `vllm` - High-speed LLM inference engine
- `transformers` - HuggingFace transformers
- `qwen_vl_utils` - Qwen VL utilities for vision processing
- `pycocotools` - COCO evaluation API
- `pandas`, `numpy` - Data processing
- `Pillow` - Image processing
- `tabulate` - Table formatting (optional, for better output display)

### Data Preparation

The ODinW dataset requires a specific directory structure:

```
/path/to/odinw_data/
├── odinw13_config.py          # Dataset configuration file (required)
├── AerialMaritimeDrone/       # Individual datasets
│   ├── large/
│   │   ├── train/
│   │   └── test/
│   └── tiled/
├── Aquarium/
├── Cottontail Rabbits/
├── EgoHands/
├── NorthAmerica Mushrooms/
├── Packages/
├── Pascal VOC/
├── Pistols/
├── Pothole/
├── Raccoon/
├── ShellfishOpenImages/
├── Thermal Dogs and People/
└── Vehicles OpenImages/
```

**Important**: The `odinw13_config.py` file must contain:
- `datasets`: List of dataset configurations
- `dataset_prefixes`: List of dataset names

## Quick Start

### 1. Inference

Run inference on the ODinW dataset using an instruct model:

```bash
bash infer_instruct.sh
```

Or customize the inference:

```bash
python run_odinw.py infer \
    --model-path /path/to/Qwen3-VL-Instruct \
    --data-dir /path/to/odinw_data \
    --output-file results/odinw_predictions.jsonl \
    --max-new-tokens 32768 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --repetition-penalty 1.0 \
    --presence-penalty 1.5
```

For thinking models with extended reasoning:

```bash
bash infer_think.sh
```

### 2. Evaluation

Evaluate the inference results using COCO metrics:

```bash
bash eval_instruct.sh
```

Or customize the evaluation:

```bash
python run_odinw.py eval \
    --data-dir /path/to/odinw_data \
    --input-file results/odinw_predictions.jsonl \
    --output-file results/odinw_eval_results.json
```

## Detailed Usage

### Inference Mode

**Basic Arguments:**
- `--model-path`: Path to the Qwen3-VL model directory (required)
- `--data-dir`: Path to ODinW data directory containing `odinw13_config.py` (required)
- `--output-file`: Path to save inference results in JSONL format (required)

**vLLM Arguments:**
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (default: auto-detect)
- `--gpu-memory-utilization`: GPU memory utilization ratio, 0.0-1.0 (default: 0.9)
- `--max-model-len`: Maximum model context length (default: 128000)
- `--max-images-per-prompt`: Maximum images per prompt (default: 10)

**Generation Parameters:**
- `--max-new-tokens`: Maximum tokens to generate (default: 32768)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Top-p sampling (default: 0.8)
- `--top-k`: Top-k sampling (default: 20)
- `--repetition-penalty`: Repetition penalty (default: 1.0)
- `--presence-penalty`: Presence penalty to reduce repetition (default: 1.5)

### Evaluation Mode

**Basic Arguments:**
- `--data-dir`: Path to ODinW data directory containing `odinw13_config.py` (required)
- `--input-file`: Inference results file in JSONL format (required)
- `--output-file`: Path to save evaluation results in JSON format (required)

## Output Files

### Inference Output

The inference script generates two files:

1. **Predictions file** (`odinw_predictions.jsonl`): JSONL file where each line contains:
```json
{
  "question_id": 0,
  "annotation": [...],
  "extra_info": {
    "dataset_name": "AerialMaritimeDrone_large",
    "img_id": 1,
    "anno_path": "/path/to/annotations.json",
    "resized_h": 640,
    "resized_w": 640,
    "img_h": 1080,
    "img_w": 1920,
    "img_path": "/path/to/image.jpg"
  },
  "result": {
    "gen": "[{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"boat\"}, ...]",
    "gen_raw": "Raw model output including thinking process"
  },
  "messages": [...]
}
```

2. **Dataset config file** (`odinw_predictions_datasets.json`): Configuration for evaluation

### Evaluation Output

The evaluation script generates a JSON file with results for each dataset:

```json
{
  "AerialMaritimeDrone_large": {
    "mAP": 0.456,
    "mAP_50": 0.678,
    "mAP_75": 0.512,
    "mAP_s": 0.234,
    "mAP_m": 0.456,
    "mAP_l": 0.567
  },
  "Aquarium_Aquarium Combined.v2-raw-1024.coco": {
    ...
  },
  ...
  "Average": 0.423
}
```

**Evaluation Metrics:**
- **mAP**: Mean Average Precision at IoU 0.5:0.95 (primary metric)
- **mAP_50**: mAP at IoU threshold 0.5
- **mAP_75**: mAP at IoU threshold 0.75
- **mAP_s**: mAP for small objects (area < 32²)
- **mAP_m**: mAP for medium objects (32² < area < 96²)
- **mAP_l**: mAP for large objects (area > 96²)

## Model-Specific Configurations

### Instruct Models (e.g., Qwen3-VL-2B-Instruct, Qwen3-VL-7B-Instruct)

Use standard parameters for balanced performance:

```bash
--max-new-tokens 32768
--temperature 0.7
--top-p 0.8
--top-k 20
--repetition-penalty 1.0
--presence-penalty 1.5
```

### Thinking Models (e.g., Qwen3-VL-2B-Thinking)

Use adjusted parameters for deeper reasoning:

```bash
--max-new-tokens 32768
--temperature 0.6
--top-p 0.95
--top-k 20
--repetition-penalty 1.0
--presence-penalty 0.0
```

**Note:** Thinking models output reasoning steps wrapped in `<think>...</think>` tags. The evaluation automatically extracts the final answer after `</think>`.

## Performance Tips

1. **GPU Memory**: Adjust `--gpu-memory-utilization` based on your GPU:
   - 0.9: Recommended for most cases
   - 0.95: For maximum throughput (may cause OOM)
   - 0.7-0.8: If experiencing OOM errors

2. **Batch Size**: vLLM automatically optimizes batch size based on available memory

3. **Tensor Parallelism**: Use `--tensor-parallel-size` for large models:
   - 2B model: 1 GPU recommended
   - 7B model: 1-2 GPUs
   - 14B+ model: 2-4 GPUs

4. **Context Length**: Reduce `--max-model-len` if memory is limited:
   - 128000: Default, works well for most cases
   - 64000: Reduces memory usage by ~40%

5. **Image Processing**: The implementation uses `smart_resize` to automatically adjust image dimensions:
   - Dimensions are made divisible by 32
   - Total pixels are constrained to [min_pixels, max_pixels]
   - Aspect ratio is preserved

## Troubleshooting

### Common Issues

**1. Config file not found**
```
FileNotFoundError: Config file not found: /path/to/odinw13_config.py
```
**Solution**: Ensure `odinw13_config.py` exists in `--data-dir`

**2. CUDA Out of Memory**
```bash
# Reduce GPU memory utilization
--gpu-memory-utilization 0.7

# Or reduce context length
--max-model-len 64000
```

**3. vLLM Multiprocessing Issues**
The code automatically sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`. If you still encounter issues:
```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

**4. Empty or Invalid JSON Output**
- Check model output format
- Verify prompt clarity
- Try adjusting temperature/top_p

**5. Low mAP Scores**
- Verify category names match dataset classes
- Check coordinate format (xyxy vs xywh)
- Ensure model outputs JSON format correctly

**6. COCO API Errors**
```
IndexError: The testing results of the whole dataset is empty.
```
**Solution**: No valid predictions were generated. Check model outputs.

## Advanced Usage

### Custom Image Resolution

Edit `dataset_utils.py` to modify resolution parameters:

```python
# Calculate image resolution parameters
patch_size = 16
merge_base = 2
pixels_per_token = patch_size * patch_size * merge_base * merge_base
min_pixels = pixels_per_token * 768
max_pixels = pixels_per_token * 12800
```

### Filtering Datasets

To evaluate only specific datasets, edit `generate_odinw_jobs()` in `dataset_utils.py`:

```python
# Only process specific datasets
dataset_filter = ['AerialMaritimeDrone', 'Aquarium']
for data_name, data_config in datasets.items():
    if data_name not in dataset_filter:
        continue
    # ... rest of the code
```

### Custom Prompt Format

Edit the prompt in `dataset_utils.py`:

```python
# Default prompt
prompt = f"Locate every instance that belongs to the following categories: '{obj_names}'. Report bbox coordinates in JSON format."

# Custom prompt example
prompt = f"Find all {obj_names} objects in the image and output their bounding boxes as JSON."
```

## Citation

If you use this code or the ODinW benchmark, please cite:

```bibtex
@inproceedings{li2022grounded,
  title={Grounded language-image pre-training},
  author={Li, Liunian Harold and Zhang, Pengchuan and Zhang, Haotian and Yang, Jianwei and Li, Chunyuan and Zhong, Yiwu and Wang, Lijuan and Yuan, Lu and Zhang, Lei and Hwang, Jenq-Neng and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10965--10975},
  year={2022}
}
```

## License

This code is released under the same license as the Qwen3-VL model.

## Support

For issues and questions:
- GitHub Issues: [Qwen3-VL Repository](https://github.com/QwenLM/Qwen3-VL)
- Documentation: See inline code comments and docstrings
