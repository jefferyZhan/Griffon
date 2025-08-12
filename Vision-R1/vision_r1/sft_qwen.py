"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys
import json

import datasets
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
import trl
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from qwen_vl_utils import process_vision_info
logger = logging.getLogger(__name__)


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )



processor = None


def convert_example(example):
    """
    correct example into "messages" 
    eg:
    {
      "system": "You are a helpful assistant.",
      "conversations": [
          {"from": "user", "value": "How many objects are included in this image?",
           "image_path": "/path/to/image.png"},
          {"from": "assistant", "value": "<think>\nI can see 10 objects\n</think>\n<answer>\n10\n</answer>"}
      ]
    }
    """
    messages = []
    # ZYF MODIFY: remove system info as not used
    # if "system" in example:
    #     messages.append({
    #         "role": "system",
    #         "content": [{"type": "text", "text": example["system"]}],
    #     })
    # else:
    #     SYSTEM_PROMPT = (
    # "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    # "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    # "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    # "<think> reasoning process here </think><answer> answer here </answer>"
    #     )
    #     messages.append({
    #         "role": "system",
    #         "content": [{"type": "text", "text": SYSTEM_PROMPT}],
    #     })

    # thinking = example.get("thinking")
    problem = example.get("problem")
    if "Examine the image for any objects from the category set. Report the coordinates of each detected object." in problem:
        pb = problem.replace("Examine the image for any objects from the category set. Report the coordinates of each detected object.", "Locate every item from the category list in the image and output the coordinates in JSON format.")
        #pb = "Please output bbox coordinates and names of every item in this image."
    elif "Can you point out" in problem:
        pb = problem.replace("Can you point out", "Locate every").replace("in the image and provide the coordinates of its location?", "in the image and output the coordinates in JSON format.")
    elif "Locate the exact position of" in problem:
        pb = problem.replace("Locate the exact position of", "Locate every").replace("in the picture, if you can.", "in the image and output the coordinates in JSON format.")
    else:
        raise AssertionError
    
    image = example.get("image")
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": pb},
            {"type": "image", "image": image},
            ]
    })
    # ZYF_MODIFY: Leave assistant blank now, due to the access to the image input width and height
    # messages.append({
    #     "role": "assistant",
    #     "content": f"{thinking}\n\n{solution}",
    # })
    
    example["messages"] = messages
    return example


def collate_fn(examples):
    texts = []
    image_inputs = []
    for example in examples:
        messages = convert_example(example)["messages"]
        imgs, vids = process_vision_info(messages)
        image_inputs.append(imgs)
        height = imgs[0].height
        width = imgs[0].width
        solution = json.loads(example.get("solution"))
        categories = [item[0] for item in solution]
        bboxes = [item[1][0] for item in solution]
        turned_bboxes = np.array(bboxes)/np.array([example.get("width"),example.get("height"),example.get("width"),example.get("height")])*np.array([width, height, width, height])
        turned_bboxes = turned_bboxes.astype(int).tolist()
        strings = [f'{{"bbox_2d": [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}], "label": "{cate}"}}' for cate, bbox in zip(categories, turned_bboxes)]
        string = ",\n\t".join(strings)
        messages.append({
            "role": "assistant",
            "content": f"```json\n[\n\t{string}\n]\n```",
        })
        texts.append(
            processor.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)
        )
    
    
    # texts = [
    #     processor.apply_chat_template( convert_example(example)["messages"], tokenize=False, add_generation_prompt=True)
    #     for example in examples
    # ]
    # image_inputs = []
    # for example in examples:
    #     imgs, vids = process_vision_info(example["messages"])
    #     image_inputs.append(imgs)
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################
    try:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    except:
        def filter_has_instance(example, minimal=1, maximum=30):
            return int(example["num_instances"])>=minimal and int(example["num_instances"])<=maximum
        dataset = load_from_disk(script_args.dataset_name)
        dataset = dataset.filter(filter_has_instance)

    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    processor.image_processor.max_pixels = training_args.max_pixels
    processor.image_processor.min_pixels = training_args.min_pixels
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # training_args.model_init_kwargs = model_kwargs
    from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
    if "Qwen2-VL" in model_args.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    elif "Qwen2.5-VL" in model_args.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    else:
        assert False, f"Model {model_args.model_name_or_path} not supported"
    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args)
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    try:
        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    except:
        pass

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["R1-V"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    #############
    # push to hub
    #############

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
        processor.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)