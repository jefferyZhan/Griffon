#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from griffon.model import *
from griffon.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from griffon.utils import auto_rank0_print

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", torch_dtype=torch.float16, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch_dtype


    auto_rank0_print(f"Loaded Griffon model: {model_path}")
    if "griffon" in model_name.lower():
        if "gemma" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            # cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = GriffonGemma2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation="eager", **kwargs)
        elif "llama2" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            # cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = GriffonLlama2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation="flash_attention_2", **kwargs)
        elif "qwen2" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = GriffonQwen25ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation="flash_attention_2", **kwargs)
    elif "llava" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    image_processor = None

    if 'llava' in model_name.lower() or 'griffon' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        if "image-token" in model_name.lower():
            model.resize_token_embeddings(len(tokenizer)-1369)
        else:
            model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        if device_map != "auto":
            vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
