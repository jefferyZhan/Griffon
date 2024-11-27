import argparse
import torch
import time

from griffon.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from griffon.conversation import conv_templates, SeparatorStyle
from griffon.model.builder import load_pretrained_model
from griffon.utils import disable_torch_init
from griffon.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from griffon.coor_utils import visualization

from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor

import requests
from PIL import Image
from io import BytesIO

import re


def extract(string, pat):
    # input: string above
    # output: [{"category_name": cls, "bbox": bbox}]
    output = []
    # This version we first 
    box_strings = string.strip().split("&")
    box_strings = [item for item in box_strings if len(pat.findall(item))]
    for b_str in box_strings:
        if "-" in b_str:
            if b_str.count("-") > 1:
                cls, bbox_str  = b_str.split("-")[-2:]
            else:
                cls, bbox_str  = b_str.split("-")
        else: # To support situation when category name is removed
            cls = "target"
            bbox_str = pat.findall(b_str)[0]
        bbox_str = bbox_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        bbox = [item for item in map(float, bbox_str.split(','))]
        # check box consistency
        if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            continue
        ins = {
            "category_name": cls,
            "bbox": bbox
        }
        output.append(ins)
    return output

def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    resize = transforms.Resize((image_processor.size["shortest_edge"],image_processor.size["shortest_edge"]))
    if getattr(model_cfg, "unshared_visual_encoder", False):
        prompt_processor = CLIPImageProcessor.from_pretrained("checkpoints/clip-vit-large-patch14")

        prompt_resize = transforms.Resize((prompt_processor.size["shortest_edge"], prompt_processor.size["shortest_edge"]))
    else:
        prompt_processor = None
        prompt_resize = None
    new_images = []
    new_prompts = []

    for i, image in enumerate(images):
        if i % 2 == 0:
            image = resize(image)
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
        else:
            if prompt_resize == None:
                image = resize(image)
                image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)
            else:
                image = prompt_resize(image)
                image = prompt_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_prompts.append(image)

    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    if len(new_prompts)>0 and all(x.shape == new_prompts[0].shape for x in new_prompts):
        new_prompts = torch.stack(new_prompts, dim=0)
    return new_images, new_prompts

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    if "llama" in model_name.lower():
        torch_dtype=torch.float16
    else:
        torch_dtype=torch.bfloat16
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device="cuda:0", torch_dtype=torch_dtype
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower() or "llama2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    elif "gemma" in model_name.lower():
        conv_mode = "gemma_instruct"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    prompt = prompt.replace("<region>","<image>")

    image_files = image_parser(args)
    images = load_images(image_files)
    images_tensor, prompt_tensors = process_images(
        images,
        image_processor,
        model.config
    )
    if torch.is_tensor(images_tensor):
        images_tensor = images_tensor.to(model.device, dtype=torch_dtype)
    else:
        images_tensor = [image.to(model.device, dtype=torch_dtype) for image in images_tensor]

    if len(prompt_tensors) > 0:
        if torch.is_tensor(prompt_tensors):
            prompt_tensors = prompt_tensors.to(model.device, dtype=torch_dtype)
        else:
            prompt_tensors = [image.to(model.device, dtype=torch_dtype) for image in prompt_tensors]
    else:
        prompt_tensors = None

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        if prompt_tensors is not None:

            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                regions=prompt_tensors,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                # return_dict_in_generate=True,
                # output_hidden_states=True
            )
        else:

            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                # return_dict_in_generate=True,
                # output_hidden_states=True
            )

    if "qwen" in args.model_path.lower() or "gemma" in args.model_path.lower():
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
    else:
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
    print(f"Output: {outputs}")
    # # print(output_ids[:, input_token_len:])
    # outputs = extract_objects(outputs)
    try:
        middle_brackets_pat = re.compile("(\[\d{1,}\.\d{3},\d{1,}\.\d{3},\d{1,}\.\d{3},\d{1,}\.\d{3}\])")
        bboxes = extract(outputs, middle_brackets_pat)
        print("Output Path: demo/{}_out.jpg".format(image_files[0].split(".")[0].replace("/","")))
        visualization(image_files[0], bboxes, "demo/{}_out.png".format(image_files[0].split(".")[0].replace("/","")))
    except:
        pass
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
