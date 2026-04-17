import os
import sys
import json
import argparse
import numpy as np
import time
from tqdm import tqdm
from typing import List, Dict, Any
from collections import defaultdict, OrderedDict
import torch

# vLLM imports
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

# pycocotools imports
from pycocotools.coco import COCO

# Local imports from refactored files
from dataset_utils import load_odinw_config, generate_odinw_jobs
from eval_utils import compute_metrics

# Set vLLM multiprocessing method
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["is_rel"] = "1"


def prepare_inputs_for_vllm(messages, processor):
    """Prepare inputs for vLLM."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # qwen_vl_utils 0.0.14+ required
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
    
    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


def run_inference(args):
    """Run inference on the ODinW dataset using vLLM."""
    print("\n" + "="*80)
    print("🚀 ODinW Inference with vLLM (High-Speed Mode)")
    print("="*80 + "\n")
    
    # Generate task list
    question_list, datasets = generate_odinw_jobs(args.data_dir, args)
    print(f"✓ Generated {len(question_list)} inference jobs\n")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Set up generation parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        stop_token_ids=[],
    )
    
    print(f"\n⚙️  Generation parameters (vLLM SamplingParams):")
    print(f"   max_tokens={sampling_params.max_tokens}")
    print(f"   temperature={sampling_params.temperature}, top_p={sampling_params.top_p}, top_k={sampling_params.top_k}")
    print(f"   repetition_penalty={sampling_params.repetition_penalty}")
    print(f"   presence_penalty={sampling_params.presence_penalty}")
    print()
    
    # Load processor
    print(f"Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("✓ Processor loaded\n")
    
    # Initialize vLLM
    print(f"Initializing vLLM with model: {args.model_path}")
    print(f"   GPU count: {torch.cuda.device_count()}")
    print(f"   Tensor parallel size: {args.tensor_parallel_size}")
    
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": args.max_images_per_prompt},
        seed=42,
    )
    print("✓ vLLM initialized successfully\n")
    
    # Prepare all inputs
    print("Preparing inputs for vLLM...")
    all_inputs = []
    
    for item in tqdm(question_list, desc="Building prompts"):
        vllm_input = prepare_inputs_for_vllm(item['messages'], processor)
        all_inputs.append(vllm_input)
    
    print(f"✓ Prepared {len(all_inputs)} inputs\n")
    
    # Batch inference
    print("="*80)
    print("🚀 Running vLLM batch inference")
    print("="*80)
    start_time = time.time()
    
    outputs = llm.generate(all_inputs, sampling_params=sampling_params)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n✓ Inference completed in {total_time:.2f} seconds")
    print(f"  Average: {total_time/len(question_list):.2f} seconds/sample")
    print(f"  Throughput: {len(question_list)/total_time:.2f} samples/second\n")
    
    # Save results
    print("Saving results...")
    results = []
    
    for idx, (item, output) in enumerate(zip(question_list, outputs)):
        response = output.outputs[0].text
        
        # Handle </think> tag
        response_final = str(response).split("</think>")[-1].strip()
        
        result = {
            "question_id": item['question_id'],
            "annotation": item['annotation'],
            "extra_info": item['extra_info'],
            "result": {"gen": response_final, "gen_raw": response},
            "messages": item['messages']
        }
        results.append(result)
    
    # Save results
    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    
    print(f"\n✓ Results saved to {args.output_file}")
    print(f"✓ Total samples processed: {len(results)}")
    
    # Save dataset config (for evaluation)
    config_output = args.output_file.replace('.jsonl', '_datasets.json')
    with open(config_output, 'w') as f:
        # Convert config for JSON serialization
        datasets_serializable = {}
        for k, v in datasets.items():
            datasets_serializable[k] = {
                'metainfo': v['metainfo'],
                'data_root': v['data_root'],
                'ann_file': v['ann_file'],
                'data_prefix': v['data_prefix']
            }
        json.dump(datasets_serializable, f, indent=2)
    print(f"✓ Dataset config saved to {config_output}")


def run_evaluation(args):
    """Run evaluation on inference results."""
    print("\n" + "="*80)
    print("🎯 ODinW Evaluation")
    print("="*80 + "\n")
    
    # Load inference results
    results = []
    with open(args.input_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    
    print(f"✓ Loaded {len(results)} inference results\n")
    
    # Load dataset config
    config_path = os.path.join(args.data_dir, "dataset_config.py")
    datasets = load_odinw_config(config_path)
    
    # Group by dataset
    all_outputs = defaultdict(list)
    for job in results:
        all_outputs[job["extra_info"]["dataset_name"]].append(job)
    
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name, sub_jobs in all_outputs.items():
        print(f"\n{'='*60}")
        print(f"Evaluating dataset: {dataset_name}")
        print(f"{'='*60}")
        
        anno_path = sub_jobs[0]["extra_info"]["anno_path"]
        coco_api = COCO(anno_path)
        
        classes = datasets[dataset_name]['metainfo']['classes']
        pred_bboxes_per_img = defaultdict(list)
        
        for job in sub_jobs:
            img_id = job["extra_info"]["img_id"]
            resized_h = job["extra_info"]["resized_h"]
            resized_w = job["extra_info"]["resized_w"]
            img_h = job["extra_info"]["img_h"]
            img_w = job["extra_info"]["img_w"]
            
            answer = job['result']['gen']
            answer = answer.replace("```json", "")
            answer = answer.replace("```", "")
            
            # Parse predictions
            import ast
            import re
            
            try:
                json_data = ast.literal_eval(answer)
                pred_bboxes = []
                pred_labels = []
                for data in json_data:
                    if len(data.get("bbox_2d", [])) != 4:
                        continue
                    pred_bboxes.append(data["bbox_2d"])
                    pred_labels.append(data["label"])
            except Exception as e:
                # If parsing fails, use empty results
                pred_bboxes = []
                pred_labels = []
            
            # Coordinate conversion (from resized to original size)
            if os.getenv("is_rel", "0") == "1":
                pred_bboxes = np.array(pred_bboxes).reshape(-1, 4) / 1000 * np.array([img_w, img_h, img_w, img_h])
            else:
                if len(pred_bboxes) > 0:
                    pred_bboxes = np.array(pred_bboxes).reshape(-1, 4) / np.array([resized_w, resized_h, resized_w, resized_h]) * np.array([img_w, img_h, img_w, img_h])
                else:
                    pred_bboxes = np.array(pred_bboxes).reshape(-1, 4)
            
            pred_bboxes = pred_bboxes.tolist()
            
            # Group by category
            pred_objs = defaultdict(list)
            for pred_bbox, pred_label in zip(pred_bboxes, pred_labels):
                pred_objs[pred_label].append(pred_bbox)
            
            for k, v in pred_objs.items():
                class_names = [name.lower() for name in classes]
                if k.lower() not in class_names:
                    continue
                pred_bboxes_per_img[img_id].append({
                    'label': class_names.index(k.lower()), 
                    'bbox': v
                })
        
        # Prepare evaluation results
        pred_results = []
        for k, v in pred_bboxes_per_img.items():
            bboxes = []
            labels = []
            for tmp in v:
                bboxes.extend(tmp['bbox'])
                labels.extend([tmp['label']] * len(tmp['bbox']))
            
            height = coco_api.imgs[k]["height"]
            width = coco_api.imgs[k]["width"]
            
            pred_tuple = (
                {'width': width, 'height': height, 'img_id': k},
                {
                    'img_id': k,
                    'bboxes': np.array(bboxes),
                    'scores': np.array([1.0] * len(bboxes)),
                    'labels': np.array(labels),
                },
            )
            pred_results.append(pred_tuple)
        
        # Compute metrics
        eval_results = compute_metrics(pred_results, _coco_api=coco_api)
        print(f"{dataset_name}: {eval_results}")
        all_results[dataset_name] = eval_results
    
    # Summarize results
    results_ordered = OrderedDict(sorted(all_results.items(), key=lambda x: x[0]))
    metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
    results_display = []
    
    for prefix, result in results_ordered.items():
        results_display.append([prefix] + [result[k] for k in metric_items])
    
    # Calculate average
    average_scores = []
    for col_idx in range(len(metric_items)):
        average_scores.append(np.mean([line[col_idx + 1] for line in results_display]))
    results_display.append(['Average'] + average_scores)
    
    # Print results table
    try:
        from tabulate import tabulate
        print("\n" + "="*80)
        print(
            tabulate(
                results_display,
                headers=["ODinW13 Dataset"] + metric_items,
                tablefmt="fancy_outline",
                floatfmt=".3f",
            )
        )
        print("="*80 + "\n")
    except ImportError:
        print("\n" + "="*80)
        print("ODinW13 Results:")
        print("="*80)
        for row in results_display:
            print(row)
        print("="*80 + "\n")
    
    # Save results
    all_results.update({"Average": average_scores[0]})
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    print(f"✓ Evaluation results saved to {args.output_file}")
    print(f"\n{'='*80}")
    print(f"Final Average mAP: {average_scores[0]:.4f}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="ODinW Evaluation with vLLM")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference with vLLM")
    infer_parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    infer_parser.add_argument("--data-dir", type=str, required=True, 
                             help="Path to ODinW data directory (containing odinw13_config.py)")
    infer_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    
    # vLLM specific parameters
    infer_parser.add_argument("--tensor-parallel-size", type=int, default=None,
                            help="Tensor parallel size (default: number of GPUs)")
    infer_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                            help="GPU memory utilization (0.0-1.0, default: 0.9)")
    infer_parser.add_argument("--max-model-len", type=int, default=128000,
                            help="Maximum model context length (default: 128000)")
    infer_parser.add_argument("--max-images-per-prompt", type=int, default=10,
                            help="Maximum images per prompt (default: 10)")
    
    # Generation parameters
    infer_parser.add_argument("--max-new-tokens", type=int, default=32768,
                            help="Maximum number of tokens to generate (default: 32768)")
    infer_parser.add_argument("--temperature", type=float, default=0.7,
                            help="Temperature for sampling (default: 0.7)")
    infer_parser.add_argument("--top-p", type=float, default=0.8,
                            help="Top-p for sampling (default: 0.8)")
    infer_parser.add_argument("--top-k", type=int, default=20,
                            help="Top-k for sampling (default: 20)")
    infer_parser.add_argument("--repetition-penalty", type=float, default=1.0,
                            help="Repetition penalty (default: 1.0)")
    infer_parser.add_argument("--presence-penalty", type=float, default=1.5,
                            help="Presence penalty (default: 1.5)")
    
    # Evaluation parser
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--data-dir", type=str, required=True,
                           help="Path to ODinW data directory (containing odinw13_config.py)")
    eval_parser.add_argument("--input-file", type=str, required=True,
                           help="Input file with inference results")
    eval_parser.add_argument("--output-file", type=str, required=True,
                           help="Output file path")
    
    args = parser.parse_args()
    
    # Automatically set tensor_parallel_size
    if args.command == 'infer' and args.tensor_parallel_size is None:
        args.tensor_parallel_size = torch.cuda.device_count()
        print(f"Auto-set tensor_parallel_size to {args.tensor_parallel_size}")
    
    if args.command == 'infer':
        run_inference(args)
    elif args.command == 'eval':
        run_evaluation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()