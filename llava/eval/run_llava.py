import argparse
import torch
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init, visualization, MultiNumberBoxFormatter
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
from torchvision import transforms

import requests
from PIL import Image
from io import BytesIO

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def expand2square_org(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, 0)) #图像始终放在左上角，原图的原点不变
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, (0, 0))
        return result

def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    qs = args.query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv_mode = "llava_llama_2"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    image_path = args.image_file
    image = load_image(image_path)
    #Process directly resize
    resize = transforms.Resize((image_processor.size["shortest_edge"],image_processor.size["shortest_edge"]))
    image = resize(image)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_dict = model.generate(
            input_ids,
            images=image_tensor,
            # choose the best type
            # do_sample=True,
            # temperature=0.2,
            num_beams=1,
            max_new_tokens=2048,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            return_dict_in_generate=True,
            output_scores=True)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_dict.sequences[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    output_ids = output_dict.sequences
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(f"{args.image_file}:",outputs)
    boxformat = MultiNumberBoxFormatter(image_processor, box_split_placeholder="&", precision=3)
    print("Box format is MultiNumberBoxFormatter")

    if not os.path.exists("inference_out/"):
        os.mkdir("inference_out/")
    if outputs.lower() == "None".lower():
        return None
    try:
        bboxes = boxformat.extract(outputs)
        visualization(image_path, bboxes, "inference_out/{}".format(image_path.split("/")[-1].split(".")[0]+"_out.jpg"))
    except:
        bboxes = boxformat.extract(f"{args.obj}-"+outputs)
        bboxes = [bboxes[0]]
        bboxes[0]["category_name"] = args.obj
        visualization(image_path, bboxes, "inference_out/{}".format(image_path.split("/")[-1].split(".")[0]+"_out.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/llava-llama-2-13b-chat-lightning-preview-griffon-final")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument('--obj', required=True, type=str, default="Target")
    args = parser.parse_args()

    eval_model(args)
