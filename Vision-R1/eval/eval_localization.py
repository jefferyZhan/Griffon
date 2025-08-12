import os, json
from typing import Dict, List, Optional, Union
import torch
import argparse
import transformers
import itertools
import copy
import re
import ast 

from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm
from functools import partial
import numpy as np
from torchvision.ops.boxes import box_area
from PIL import Image
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from torchvision import transforms
from datetime import timedelta

from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import json
import random
import io
import ast 

# griffon
from griffon.coor_utils import xyxy2xywh, accum_probs, visualization, xywh2xyxy

from griffon.eval.run_griffon import load_image, extract
from griffon.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from griffon.conversation import conv_templates, SeparatorStyle
from griffon.model.builder import load_pretrained_model
from griffon.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from griffon.utils import auto_rank0_print

#qwen
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info

#internvl
# import torchvision.transforms as T 
# from torchvision.transforms.functional import InterpolationMode
# from transformers import AutoModel, AutoTokenizer

# ferret
# from ferret.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# from ferret.model.builder import load_pretrained_model
# from ferret.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from ferret.conversation import conv_templates, SeparatorStyle


COCO_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', \
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', \
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', \
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', \
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', \
              'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', \
              'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', \
              'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VOCAB_IMAGE_W = 1000
VOCAB_IMAGE_H = 1000

DATASET_MAP = {
    "AerialMaritimeDrone": "dataset/odinw/data/AerialMaritimeDrone/large/valid/annotations_without_background.json",
    "Aquarium": "dataset/odinw/data/Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/annotations_without_background.json",
    "CottontailRabbits": "dataset/odinw/data/CottontailRabbits/valid/annotations_without_background.json",
    "EgoHands": "dataset/odinw/data/EgoHands/generic/valid/annotations_without_background.json",
    "NorthAmericaMushrooms": "dataset/odinw/data/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/annotations_without_background.json",
    "Packages": "dataset/odinw/data/Packages/Raw/valid/annotations_without_background.json",
    "PascalVOC": "dataset/odinw/data/PascalVOC/valid/annotations_without_background.json",
    "pistols": "dataset/odinw/data/pistols/export/val_annotations_without_background.json",
    "pothole": "dataset/odinw/data/pothole/valid/annotations_without_background.json",
    "Raccoon": "dataset/odinw/data/Raccoon/Raccoon.v2-raw.coco/valid/annotations_without_background.json",
    "ShellfishOpenImages": "dataset/odinw/data/ShellfishOpenImages/raw/valid/annotations_without_background.json",
    "thermalDogsAndPeople": "dataset/odinw/data/thermalDogsAndPeople/valid/annotations_without_background.json",
    "VehiclesOpenImages": "dataset/odinw/data/VehiclesOpenImages/416x416/valid/annotations_without_background.json",
    "coco2017": "dataset/coco2017/annotations/instances_val2017.json",
    "boggleBoards": "dataset/odinw/data/boggleBoards/416x416AutoOrient/export/val_annotations_without_background.json",
    "ChessPieces": "dataset/odinw/data/ChessPieces/Chess Pieces.v23-raw.coco/valid/annotations_without_background.json",
    "syntheticFruit": "dataset/odinw/data/syntheticFruit/valid/annotations_without_background.json",
    "websiteScreenshots": "dataset/odinw/data/websiteScreenshots/valid/annotations_without_background.json",
    "openPoetryVision": "dataset/odinw/data/openPoetryVision/512x512/valid/annotations_without_background.json",
    "ThermalCheetah": "dataset/odinw/data/ThermalCheetah/valid/annotations_without_background.json",
    "WildfireSmoke": "dataset/odinw/data/WildfireSmoke/valid/annotations_without_background.json",
    "MaskWearing": "dataset/odinw/data/MaskWearing/raw/valid/annotations_without_background.json",
    "OxfordPets_by_species": "dataset/odinw/data/OxfordPets/by-species/valid/annotations_without_background.json",
    "AerialMaritimeDrone_Tile": "dataset/odinw/data/AerialMaritimeDrone/tiled/valid/annotations_without_background.json",
    "MountainDewCommercial": "dataset/odinw/data/MountainDewCommercial/valid/annotations_without_background.json",
    "selfdrivingCar": "dataset/odinw/data/selfdrivingCar/fixedLarge/export/val_annotations_without_background.json",
    "vector": "dataset/odinw/data/vector/vectorCompleteDataset.v5-v5-(resize-+-grayscale-+-rotate-+-blur-+-bb-optimizations).coco/valid/annotations_without_background.json",
    "spec_hand": "dataset/odinw/data/EgoHands/specific/valid/annotations_without_background.json"
    }
IMAGE_FOLDER_MAP = {
    "AerialMaritimeDrone": "dataset/odinw/data/AerialMaritimeDrone/large/valid",
    "Aquarium": "dataset/odinw/data/Aquarium/Aquarium Combined.v2-raw-1024.coco/valid",
    "CottontailRabbits": "dataset/odinw/data/CottontailRabbits/valid",
    "EgoHands": "dataset/odinw/data/EgoHands/generic/valid",
    "NorthAmericaMushrooms": "dataset/odinw/data/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid",
    "Packages": "dataset/odinw/data/Packages/Raw/valid",
    "PascalVOC": "dataset/odinw/data/PascalVOC/valid",
    "pistols": "dataset/odinw/data/pistols/export",
    "pothole": "dataset/odinw/data/pothole/valid",
    "Raccoon": "dataset/odinw/data/Raccoon/Raccoon.v2-raw.coco/valid",
    "ShellfishOpenImages": "dataset/odinw/data/ShellfishOpenImages/raw/valid",
    "thermalDogsAndPeople": "dataset/odinw/data/thermalDogsAndPeople/valid",
    "VehiclesOpenImages": "dataset/odinw/data/VehiclesOpenImages/416x416/valid",
    "coco2017": "dataset/coco2017/val2017",
    "boggleBoards": "dataset/odinw/data/boggleBoards/416x416AutoOrient/export",
    "ChessPieces": "dataset/odinw/data/ChessPieces/Chess Pieces.v23-raw.coco/valid/",
    "syntheticFruit": "dataset/odinw/data/syntheticFruit/valid/",
    "websiteScreenshots": "dataset/odinw/data/websiteScreenshots/valid/",
    "openPoetryVision": "dataset/odinw/data/openPoetryVision/512x512/valid/",
    "ThermalCheetah": "dataset/odinw/data/ThermalCheetah/valid/",
    "WildfireSmoke": "dataset/odinw/data/WildfireSmoke/valid",
    "MaskWearing": "dataset/odinw/data/MaskWearing/raw/valid",
    "OxfordPets_by_species": "dataset/odinw/data/OxfordPets/by-species/valid",
    "AerialMaritimeDrone_Tile": "dataset/odinw/data/AerialMaritimeDrone/tiled/valid/",
    "MountainDewCommercial": "dataset/odinw/data/MountainDewCommercial/valid",
    "selfdrivingCar": "dataset/odinw/data/selfdrivingCar/fixedLarge/export/",
    "vector": "dataset/odinw/data/vector/vectorCompleteDataset.v5-v5-(resize-+-grayscale-+-rotate-+-blur-+-bb-optimizations).coco/valid/",
    "spec_hand": "dataset/odinw/data/EgoHands/specific/valid"
}

def auto_rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def xyxy2xywh(bbox):
    # top left and bottom right 2 top left and b
    x = np.asarray(bbox)
    y = np.zeros_like(x)
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = (x[:, 2] - x[:, 0])
    y[:, 3] = (x[:, 3] - x[:, 1])
    return y.tolist()

def visualization(image_path, extract_bboxes, save_path, box_pattern=0):
    # 打开图片
    if isinstance(image_path, str):
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
    else:
        image = image_path
        draw = ImageDraw.Draw(image)
    height = image.height
    width = image.width

    # 提取线宽和字体大小
    line_width = int(width * 0.005) if width * 0.005 > 2 else 2  # 确保线宽至少为2像素
    font_size = int(height * 0.025) if height * 0.025 > 15 else 15  # 确保字体大小至少为15像素
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)

    # 提取框和需要标注的类别
    if isinstance(extract_bboxes[0], dict):
        bboxes = [ex["bbox"] for ex in extract_bboxes]
        classes = [ex["category_name"] for ex in extract_bboxes]
        bboxes = np.asarray(bboxes)
        bboxes[:, ::2] *= width
        bboxes[:, 1::2] *= height
        bboxes = bboxes.tolist()
    else:
        bboxes = extract_bboxes[0]
        bboxes = np.asarray(bboxes)
        bboxes[:, ::2] *= width
        bboxes[:, 1::2] *= height
        bboxes = bboxes.tolist()
        classes = len(bboxes)*["EXP"]
    
    # if box_pattern != 0:
    #     if box_pattern == 1:
    #         bboxes = cxcywh2xyxy(bboxes)

    for i, bbox in enumerate(bboxes):
        try:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            draw.rectangle(bbox, outline=color, width=line_width)

            text = classes[i]
            # text_w, text_h = draw.textsize(text, font=font)
            left, top, right, bottom = draw.textbbox((bbox[0], bbox[1]), text, font)
            text_w, text_h = right - left, bottom - top

            draw.rectangle([bbox[0], bbox[1], bbox[0]+text_w, bbox[1]+text_h], fill=color)
            draw.text((bbox[0], bbox[1]-line_width), text, fill=(255, 255, 255), font=font)
        except:
            color = random.randint(0, 255)

            draw.rectangle(bbox, outline=color, width=line_width)

            text = classes[i]
            # text_w, text_h = draw.textsize(text, font=font)
            left, top, right, bottom = draw.textbbox((bbox[0], bbox[1]), text, font)
            text_w, text_h = right - left, bottom - top
            draw.rectangle([bbox[0], bbox[1], bbox[0]+text_w, bbox[1]+text_h], fill=color)
            draw.text((bbox[0], bbox[1]-line_width), text, fill=255, font=font)
    
    try:
        image.save(save_path)
    except:
        print("Error: cannot save the output image.")
        pass

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def parse_rec_output(query, text):
    cate = query.split('<ref>')[-1].split('</ref>')[0]
    PATTERN = re.compile(r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*')
    predict_bbox = re.findall(PATTERN, text)
    try:
        predict_bbox = (float(predict_bbox[0][0]), float(predict_bbox[0][1]), float(predict_bbox[0][2]),
                        float(predict_bbox[0][3]))
    except:
        predict_bbox = (0., 0., 0., 0.)
    return [{'label': cate, 'bbox_2d': predict_bbox}]

def resize_bbox(box, image_w=None, image_h=None):
    ratio_w =  1.0 / VOCAB_IMAGE_W
    ratio_h =  1.0 / VOCAB_IMAGE_H

    new_box = [box[0] * ratio_w, box[1] * ratio_h, \
               box[2] * ratio_w, box[3] * ratio_h]
    return new_box

def decode_bbox_from_caption(query, text, img_w, img_h, verbose=False):
    cate = query.split('What are the locations of')[-1].strip().split('?')[0].strip()
    entities = []
    boxes = []
    
    start = 0
    in_brackets = False
    entity = ""
    box = ""
    
    for i, char in enumerate(text):
        if char == '[':
            in_brackets = True
            entity = text[start:i].strip()
            start = i + 1
        elif char == ']':
            in_brackets = False
            box = text[start:i].strip()
            start = i + 1
            
            # Convert box string to list of integers
            box_list = list(map(int, box.split(',')))
            resized_box_list = resize_bbox(box_list, img_w, img_h)
            entities.append(entity)
            boxes.append(resized_box_list)
            
            # Skip until the next entity (ignoring periods or other delimiters)
            while start < len(text) and text[start] not in ['.', ',', ';', '!', '?']:
                start += 1
            start += 1  # Skip the delimiter
    entities = [cate]*len(boxes)
        
    return entities, boxes

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def plot_bounding_boxes(im, bounding_boxes, input_width, input_height):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    #font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    try:
      json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
      end_idx = bounding_boxes.rfind('"}') + len('"}')
      truncated_text = bounding_boxes[:end_idx] + "]"
      json_output = ast.literal_eval(truncated_text)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
      # Select a color from the list
      color = colors[i % len(colors)]

      # Convert normalized coordinates to absolute coordinates
      abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
      abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
      abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
      abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)

      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1
      print((abs_x1, abs_y1), (abs_x2, abs_y2))

      # Draw the bounding box
      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
      )

      # Draw the text
      if "label" in bounding_box:
        draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color)

    # Display the image
    img.save("./test_qwen.jpg")

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

class EVALDataset(Dataset):
    def __init__(self, data_path: str, prompt: str, image_folder: str):
        super(EVALDataset, self).__init__()
        f = open(data_path, "r", encoding="utf-8")
        whole_annotations = json.load(f)
        dict_images = whole_annotations["images"]
        catid2name = {cate["id"]: cate["name"] for cate in whole_annotations["categories"]}
        self.list_data_dict = dict_images
        if len(catid2name.values()) > 1:
            if len(catid2name.values()) >= 80:
                input_cate = catid2name.values()
            else:
                diff = list(set(COCO_NAMES) - set(catid2name.values()))
                input_cate = list(catid2name.values()) + diff[:(80-len(catid2name.values()))]
            self.prompt = prompt.replace("<category set>", ", ".join(input_cate))
        else:
            self.prompt = "Can you point out {} in the image and provide the coordinates of its location?".format(list(catid2name.values())[0].split("-")[-1])
        # self.prompt = prompt
        self.image_folder = image_folder
        

    def __len__(self):
        return len(self.list_data_dict)

    def get_item(self, index):
        """
            return {
                "image_path":
                "height":
                "width":
            }
        """
        source = self.list_data_dict[index] # an item in coco style image
        img_path = source["file_name"]
        img_id = source["id"]
        width = source["width"]
        height = source["height"]
        ret = {
            "image_path": img_path,
            "query": self.prompt, 
            "height": height,
            "width": width,
            "image_id": img_id,
        }
        return ret
    
    def __getitem__(self, index):
        source = self.get_item(index)
        image_path = os.path.join(self.image_folder, source["image_path"])
        ret = {
            # "msg": messages,
            "query": source['query'],
            "height": source["height"],
            "width": source["width"],
            "image_name": source['image_path'],
            "image_path": image_path,
            "image_id": source["image_id"],
        }
        return ret

class SINGLEEVALDataset(Dataset):
    def __init__(self, data_path: str, prompt: str, image_folder: str):
        super(SINGLEEVALDataset, self).__init__()
        f = open(data_path, "r", encoding="utf-8")
        whole_annotations = json.load(f)
        dict_images = whole_annotations["images"]
        catid2name = {cate["id"]: cate["name"] for cate in whole_annotations["categories"]}
        self.list_data_dict = [(image_info, cate) for image_info in dict_images for cate in catid2name.values()]
        self.prompt = prompt
        self.image_folder = image_folder
        

    def __len__(self):
        return len(self.list_data_dict)

    def get_item(self, index):
        """
            return {
                "image_path":
                "height":
                "width":
            }
        """
        source, cate = self.list_data_dict[index] # an item in coco style image
        img_path = source["file_name"]
        img_id = source["id"]
        width = source["width"]
        height = source["height"]
        
        ret = {
            "image_path": img_path,
            "query": self.prompt.replace("<category set>", cate), 
            "height": height,
            "width": width,
            "image_id": img_id,
        }
        if index == 0:
            print(ret["query"])
        return ret
    
    def __getitem__(self, index):
        source = self.get_item(index)
        image_path = os.path.join(self.image_folder, source["image_path"])
        ret = {
            # "msg": messages,
            "query": source['query'],
            "height": source["height"],
            "width": source["width"],
            "image_path": image_path,
            "image_id": source["image_id"],
        }
        return ret

class SINGLE_POS_EVALDataset(Dataset):
    def __init__(self, data_path: str, prompt: str, image_folder: str):
        super(SINGLE_POS_EVALDataset, self).__init__()
        coco = COCO(data_path)

        # 获取所有类别信息
        categories = coco.loadCats(coco.getCatIds())
        f = open(data_path, "r", encoding="utf-8")
        whole_annotations = json.load(f)
        dict_images = whole_annotations["images"]
        catid2name = {cate["id"]: cate["name"] for cate in whole_annotations["categories"]}
        image_cate_list = []
        for image_info in dict_images:
            image_id = image_info['id']
            annotation_ids = coco.getAnnIds(imgIds=image_id)
            annotations = coco.loadAnns(annotation_ids)
            category_ids = set([ann['category_id'] for ann in annotations])
            image_category_names = [cat['name'] for cat in categories if cat['id'] in category_ids]
            for cat_name in image_category_names:
                image_cate_list.append((image_info, cat_name))
        self.list_data_dict = image_cate_list
        self.prompt = prompt
        self.image_folder = image_folder
        

    def __len__(self):
        return len(self.list_data_dict)

    def get_item(self, index):
        """
            return {
                "image_path":
                "height":
                "width":
            }
        """
        source, cate = self.list_data_dict[index] # an item in coco style image
        img_path = source["file_name"]
        img_id = source["id"]
        width = source["width"]
        height = source["height"]
        
        ret = {
            "image_path": img_path,
            "query": self.prompt.replace("<category set>", cate), 
            "height": height,
            "width": width,
            "image_id": img_id,
        }
        if index == 0:
            print(ret["query"])
        return ret
    
    def __getitem__(self, index):
        source = self.get_item(index)
        image_path = os.path.join(self.image_folder, source["image_path"])
        ret = {
            "query": source['query'],
            "height": source["height"],
            "width": source["width"],
            "image_name": source['image_path'],
            "image_path": image_path,
            "image_id": source["image_id"],
        }
        return ret


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def collate_fn_qwen(batches, processor):
    # dataloader会将数据自动转为cuda
    msgs = []
    for _ in batches:
        msg = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": _["image_path"],
                    },
                    {"type": "text", "text": _["query"]},
                ],
            }
        ]
        msgs.append(msg)
    infos = []

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in msgs
    ]
    image_inputs, video_inputs = process_vision_info(msgs)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    for i, _ in enumerate(batches):
        input_height = inputs['image_grid_thw'][i][1]*14
        input_width = inputs['image_grid_thw'][i][2]*14
        info = {"height": _["height"], "width": _["width"], "image_id": _["image_id"], "input_height": input_height, "input_width": input_width, "image_name": _["image_name"], "image_path": _["image_path"], 'query': _['query']}
        infos.append(info)

    # TODO: Support batch size>1
    return inputs, infos

def padding_left(input_ids):
    seq_length = torch.tensor([input_id.shape[0] for input_id in input_ids])
    max_length = seq_length.max()
    pad_length = max_length - seq_length
    x = torch.zeros((len(input_ids), max_length), dtype=input_ids[0].dtype)
    for i, input_id in enumerate(input_ids):
        x[i, pad_length[i]:] = input_id
    return x

def collate_fn_internvl(batches, tokenizer, dynamic_image_size, use_thumbnail, input_size):
    pixel_values_list = []
    _transform = build_transform(input_size=input_size)
    for _ in batches:
        image_path = _['image_path']
        image = Image.open(image_path).convert('RGB')
        if dynamic_image_size:
            images = dynamic_preprocess(image, image_size=input_size,
                                        use_thumbnail=use_thumbnail,
                                        max_num=6)
        else:
            images = [image]
        pixel_values = [_transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list, dim=0)
    texts = [_['query'] for _ in batches]
    infos = []
    for _ in batches:
        info = {"height": _["height"], "width": _["width"], "image_id": _["image_id"], "image_name": _["image_name"], "image_path": _["image_path"], "query": _["query"]}
        infos.append(info)

    return pixel_values, texts, infos


def collate_fn_ferret(batches, tokenizer, conv_mode, image_processor):
    # dataloader会将数据自动转为cuda
    image_tensors = []
    for _ in batches:
        image_path = _['image_path']
        img = load_image(image_path)
            
        img_tensor = image_processor.preprocess(img, return_tensors='pt', do_resize=True, 
                                                    do_center_crop=False, size=[336, 336])['pixel_values']
        image_tensors.append(img_tensor)

    input_ids = []
    infos = []
    for i,_ in enumerate(batches):
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], _["query"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids.append(tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'))
        info = {"query": _["query"], "height": _["height"], "width": _["width"], "image_id": _["image_id"], "image_name": _["image_name"], "image_path":_["image_path"]}
        infos.append(info)

    if len(batches) > 1:
        input_ids = padding_left(input_ids)
        return input_ids, torch.cat(image_tensors, dim=0), infos
    else:
        return input_ids[0].unsqueeze(0), image_tensors[0], infos

def collate_fn_griffon(batches, tokenizer, conv_mode, image_processor):
    # dataloader会将数据自动转为cuda
    image_tensors = []
    _resize = transforms.Resize((image_processor.size["shortest_edge"], image_processor.size["shortest_edge"]))
    for _ in batches:
        image_path = _['image_path']
        img = load_image(image_path)
        img = _resize(img)
        img_tensor = image_processor.preprocess(img, return_tensors='pt')['pixel_values']
        image_tensors.append(img_tensor)

    input_ids = []
    infos = []
    for _ in batches:
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], _["query"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids.append(tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'))
        info = {"query": _["query"], "height": _["height"], "width": _["width"], "image_id": _["image_id"], "image_name":_["image_name"], "image_path":_["image_path"]}
        infos.append(info)

    if len(batches) > 1:
        input_ids = padding_left(input_ids)
        return input_ids, torch.cat(image_tensors, dim=0), infos
    else:
        return input_ids[0].unsqueeze(0), image_tensors[0], infos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="JefferyZhan/Griffon-G-gemma2-9b")
    parser.add_argument("--model-type", type=str, default="griffon", choices=["qwen", "internvl", "ferret", "griffon"], required=True)
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--query", type=str, required=True) 
    parser.add_argument('--dataset', type=str, help="Path to coco2017 val annotations", required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--output-path', type=str, default="./eval_output/detection")
    parser.add_argument('--init',type=str, default="tcp://127.0.0.1:12457")
    parser.add_argument("--single", action="store_true", help="Set for ODINW Evaluation with visual grounding setting")
    parser.add_argument("--pos", action="store_true", help="Set for ODINW Evaluation with postive categories following the Qwen2.5VL Setting")
    parser.add_argument("--max-new-tokens", default=1024, type=int, help="Max new tokens to be generated")
    args = parser.parse_args()

    # Env Init
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
        init_method=str(args.init),
        timeout = timedelta(minutes=60)
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    #Load Model & Model init

    model_path = args.model_path
    conv_mode = "llava_llama_2"
    #args.model_path = model_path
    if args.model_type == 'qwen':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(model_path,use_fast=True,min_pixels=28*28,max_pixels=12800*28*28) #padding_side='left'
    elif args.model_type == 'internvl':
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            # assign=True
            ).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
        input_size = model.config.force_image_size or model.config.vision_config.image_size
        use_thumbnail = model.config.use_thumbnail
    elif args.model_type == 'ferret':
        conv_mode = "ferret_v1"
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name)
    elif args.model_type == "griffon":
        conv_mode = "llava_llama_2"
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    else:
        raise NotImplementedError

    prompt = args.query
    if args.model_type == 'griffon' or args.model_type == 'ferret':
        if model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    
    model_out_dir = os.path.join(args.output_path, args.model_path.split("/")[-1])
    #import pdb; pdb.set_trace()
    os.makedirs(model_out_dir, exist_ok=True)

    datasets = {ds:DATASET_MAP[ds] for ds in args.dataset.split(",")}

    for ds, dataset_name in datasets.items():
        model_data_out_dir = os.path.join(model_out_dir, ds)
        # model_data_out_dir = os.path.join(model_out_dir, 'pos_query')
        # model_data_out_dir = os.path.join(model_data_out_dir, ds)
        image_folder = IMAGE_FOLDER_MAP[ds]
        os.makedirs(model_data_out_dir, exist_ok=True)
        dataset_path = dataset_name
        whole_annotations = json.load(open(dataset_path, "r"))
        catname2id = {cate["name"].lower():cate["id"] for cate in whole_annotations["categories"]}
        if args.single:
            if args.pos:
                dataset = SINGLE_POS_EVALDataset(dataset_path, prompt, image_folder)
            else:
                dataset = SINGLEEVALDataset(dataset_path, prompt, image_folder)
        else:
            dataset = EVALDataset(dataset_path, prompt, image_folder)
        auto_rank0_print("{} input prompt is {}".format(ds, dataset.prompt))

        conv = conv_templates[conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        #import pdb; pdb.set_trace()
        if args.model_type == 'qwen':
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                sampler = InferenceSampler(len(dataset)),
                batch_size = args.batch_size,
                num_workers = args.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=partial(collate_fn_qwen, processor=processor)
            )
        elif args.model_type == 'internvl':
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                sampler = InferenceSampler(len(dataset)),
                batch_size = args.batch_size,
                num_workers = args.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=partial(collate_fn_internvl, tokenizer=tokenizer, dynamic_image_size=True, use_thumbnail=use_thumbnail, input_size=input_size)
            )
        elif args.model_type == 'ferret':
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                sampler = InferenceSampler(len(dataset)),
                batch_size = args.batch_size,
                num_workers = args.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=partial(collate_fn_ferret, tokenizer=tokenizer, conv_mode=conv_mode, image_processor=image_processor)
            )
        elif args.model_type == 'griffon':
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                sampler = InferenceSampler(len(dataset)),
                batch_size = args.batch_size,
                num_workers = args.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=partial(collate_fn_griffon, tokenizer=tokenizer, conv_mode=conv_mode, image_processor=image_processor)
            )
        else:
            raise NotImplementedError

        if not os.path.exists(os.path.join(model_data_out_dir,"original_pred.pth")):
            eval_outputs = []
            with torch.inference_mode():
                if args.model_type == 'qwen':
                    for i, (inputs, infos) in tqdm(enumerate(dataloader)):
                        # import pdb; pdb.set_trace()
                        assert args.batch_size == 1
                        inputs = inputs.to(model.device)
                        generated_ids = model.generate(**inputs, use_cache=True, do_sample=False, max_new_tokens=512)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        output_text = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                    
                        for info, output in zip(infos, output_text):
                            eval_outputs.append({
                                "output": output,
                                "query": info['query'],
                                "image_id": info["image_id"],
                                "image_name": info["image_name"],
                                "image_path": info["image_path"],
                                "height": info["height"],
                                "input_height": info["input_height"],
                                "input_width": info["input_width"],
                                "width": info["width"],
                            })
                elif args.model_type == 'internvl':
                    for i, (pixel_values, questions, infos) in tqdm(enumerate(dataloader)):
                        assert args.batch_size == 1
                        pixel_values = pixel_values.to(torch.bfloat16).cuda()
                        generation_config = dict(
                            num_beams=1,
                            max_new_tokens=1024,
                            min_new_tokens=1,
                            do_sample=True if args.temperature > 0 else False,
                            temperature=args.temperature,
                        )
                        pred = model.chat(
                            tokenizer=tokenizer,
                            pixel_values=pixel_values,
                            question=questions[0],
                            generation_config=generation_config,
                            verbose=True
                        )
                        output_text = [pred]
                        for info, output in zip(infos, output_text):
                            eval_outputs.append({
                                "output": output,
                                "query": info["query"],
                                "image_id": info["image_id"],
                                "height": info["height"],
                                "width": info["width"],
                                "image_name": info["image_name"],
                                "image_path": info["image_path"]
                            })
                elif args.model_type == 'ferret':
                    for i, (input_ids, image_tensors, infos) in tqdm(enumerate(dataloader)):
                        input_ids = input_ids.cuda()
                        image_tensors = image_tensors.half().cuda()
                        assert args.batch_size == 1
                        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                        output_ids = model.generate(
                            input_ids,
                            images=image_tensors,
                            do_sample=True,
                            temperature=0.001,
                            top_p=None,
                            num_beams=1,
                            max_new_tokens=1024,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria]
                        )
                        input_token_len = input_ids.shape[1]
                        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                        if n_diff_input_output > 0:
                            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:].cpu(), skip_special_tokens=True)
                        outputs = [output.strip()[:-len(stop_str)] if output.endswith(stop_str) else output.strip() for output in outputs]
                    
                        for info, output in zip(infos, [outputs]):
                            eval_outputs.append({
                                "output": output,
                                "image_id": info["image_id"],
                                "height": info["height"],
                                "width": info["width"],
                                "image_name": info["image_name"],
                                "image_path": info["image_path"],
                                "query": info["query"]
                            })
                elif args.model_type == 'griffon':
                    for i, (input_ids, image_tensors, infos) in tqdm(enumerate(dataloader)):
                        input_ids = input_ids.cuda()
                        image_tensors = image_tensors.half().cuda()
                        assert args.batch_size == 1
                        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                        output_dict = model.generate(
                            input_ids,
                            images=image_tensors.to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
                            do_sample=False,
                            # temperature=0.2,
                            num_beams=1,
                            max_new_tokens=2048,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                            return_dict_in_generate=True,
                            output_scores=True
                        )
                        output_ids = output_dict.sequences
                        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                        outputs = outputs.strip()

                        try:
                            # only for llama2 model
                            probs = accum_probs(output_ids, torch.cat(output_dict.scores).softmax(-1),start_token=29987, end_token=29899)
                        except:
                            print(outputs)
                            print(infos)
                            raise AssertionError

                        for info, output, pbs in zip(infos, [outputs], [probs]):
                            eval_outputs.append({
                                "output": output,
                                "image_id": info["image_id"],
                                "height": info["height"],
                                "width": info["width"],
                                "scores": pbs,
                                "image_name": info["image_name"],
                                "image_path": info["image_path"],
                                "query": info["query"]
                            })
                else:
                    raise NotImplementedError

                torch.distributed.barrier()
                world_size = torch.distributed.get_world_size()
                merged_outputs = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(merged_outputs, eval_outputs)
                merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
                if torch.distributed.get_rank() == 0:
                    torch.save(merged_outputs, os.path.join(model_data_out_dir,"original_pred.pth"))
                    print(f"Begin to eval {ds}")
                    print("Begin to eval")

                    eval_list = []
                    if args.model_type == 'qwen':
                        for j, output in tqdm(enumerate(merged_outputs)):
                            try:
                                bbox = output["output"]
                                extracted = parse_json(bbox)
                                
                                try:                                                                                                                                                                         
                                    extracted = ast.literal_eval(extracted)                                                                                                                                                           
                                except Exception as e:                                                                                           
                                    end_idx = extracted.rfind('"}') + len('"}')                                                                                                                                           
                                    truncated_text = extracted[:end_idx] + "]"                                               
                                    extracted = ast.literal_eval(truncated_text)
                                for idx, item in enumerate(extracted):
                                    #import pdb; pdb.set_trace()
                                    try:
                                        predict_bbox = torch.tensor(xyxy2xywh(item["bbox_2d"]), dtype=torch.float32).view(-1, 4)
                                        predict_bbox[:, ::2] = predict_bbox[:, ::2] / output["input_width"] * output["width"]
                                        predict_bbox[:, 1::2] = predict_bbox[:, 1::2] / output["input_height"] * output["height"]

                                        predict_id = catname2id[item["label"].lower()]
                                        ret = {
                                            "image_id": output["image_id"],
                                            "category_id": predict_id,
                                            "bbox": predict_bbox.tolist()[0],
                                            "score": 0.99,
                                        }
                                        eval_list.append(ret)
                                    except:
                                        continue
                                visual_save_path = os.path.join(model_data_out_dir, "detection/{}_{}".format(j,output["image_name"].split("/")[-1]))
                                visual_save_dir = os.path.dirname(visual_save_path)
                                if not os.path.isdir(visual_save_dir):
                                    os.makedirs(visual_save_dir)
                                try:
                                    _extracted = []
                                    for extract_item in extracted:
                                        ori_bbox = extract_item['bbox_2d']
                                        resized_bbox = [ori_bbox[0]/output["input_width"], ori_bbox[1]/output["input_height"], ori_bbox[2]/output["input_width"], ori_bbox[3]/output["input_height"]]
                                        _extract_item  = {'category_name': extract_item['label'], 'bbox': resized_bbox}
                                        _extracted.append(_extract_item)
                                
                                    visualization(os.path.join(image_folder, output["image_name"]), _extracted, visual_save_path)
                                except Exception as e:
                                    print(e)
                                    print(output["query"])
                                    print(output["output"])
                                    print(extracted)
                            except Exception as e:
                                print(e)
                                print(output["query"])
                                print(output["output"])
                                print(extracted)
                                continue
                    elif args.model_type == 'internvl':
                        for j, output in enumerate(merged_outputs):
                            try:
                                bbox = output["output"]
                                try:
                                    extracted = parse_json(bbox)
                                    
                                    try:                                                                                                                                                                         
                                        extracted = ast.literal_eval(extracted)                                                                                                                                                           
                                    except Exception as e:                                                                                           
                                        end_idx = extracted.rfind('"}') + len('"}')                                                                                                                                           
                                        truncated_text = extracted[:end_idx] + "]"                                               
                                        extracted = ast.literal_eval(truncated_text)
                                except:
                                    extracted = parse_rec_output(output['query'], output['output'])
                                    # print(extracted)
                                for idx, item in enumerate(extracted):
                                    #import pdb; pdb.set_trace()
                                    try:
                                        predict_bbox = torch.tensor(xyxy2xywh(item["bbox_2d"]), dtype=torch.float32).view(-1, 4)
                                        predict_bbox[:, ::2] = predict_bbox[:, ::2] / 1000 * output["width"]
                                        predict_bbox[:, 1::2] = predict_bbox[:, 1::2] / 1000 * output["height"]

                                        predict_id = catname2id[item["label"].lower()]
                                        ret = {
                                            "image_id": output["image_id"],
                                            "category_id": predict_id,
                                            "bbox": predict_bbox.tolist()[0],
                                            "score": 0.99,
                                        }
                                        eval_list.append(ret)
                                    except Exception as e:
                                        print(e)
                                        print(output["query"])
                                        print(output["output"])
                                        print(extracted)
                                        continue
                                visual_save_path = os.path.join(model_data_out_dir, "detection/{}_{}".format(j,output["image_name"].split("/")[-1]))
                                visual_save_dir = os.path.dirname(visual_save_path)
                                if not os.path.isdir(visual_save_dir):
                                    os.makedirs(visual_save_dir)
                                try:
                                    _extracted = []
                                    for extract_item in extracted:
                                        _extract_item  = {'category_name': extract_item['label'], 'bbox': [x/1000.0 for x in extract_item['bbox_2d']]}
                                        _extracted.append(_extract_item)
                                
                                    visualization(os.path.join(image_folder,output["image_name"]), _extracted, visual_save_path)
                                except Exception as e:
                                    print(e)
                                    print(output["query"])
                                    print(output["output"])
                                    print(extracted)    
                            except Exception as e:
                                print(e)
                                print(output["query"])
                                print(output["output"])
                                continue
                    elif args.model_type == 'ferret':
                        for j, output in enumerate(merged_outputs):
                            try:
                                bbox = output["output"]
                                cates, bboxes = decode_bbox_from_caption(output['query'], bbox[0], output["width"], output["height"])
                                extracted = []
                                for idx, (item_cate, item_bbox) in enumerate(zip(cates, bboxes)):
                                    try:
                                        predict_bbox = torch.tensor(xyxy2xywh(item_bbox), dtype=torch.float32).view(-1, 4)
                                        predict_bbox[:, ::2] *= output["width"]
                                        predict_bbox[:, 1::2] *= output["height"]
                                        predict_cate = item_cate
                                        # if predict_cate.lower() == 'rabbit' or predict_cate.lower() == 'bunny':
                                        #     predict_cate = 'Cottontail_Rabbit'
                                        predict_id = catname2id[predict_cate]
                                        ret = {
                                            "image_id": output["image_id"],
                                            "category_id": predict_id,
                                            "bbox": predict_bbox.tolist()[0],
                                            "score": 0.99,
                                        }
                                        eval_list.append(ret)
                                        extracted.append({'category_name': item_cate, 'bbox': item_bbox})
                                    except Exception as e:
                                        print(e)
                                        print(output["query"])
                                        print(output["output"])
                                        print(zip(cates, bboxes))
                                        continue
                                visual_save_path = os.path.join(model_data_out_dir, "detection/{}_{}".format(j,output["image_name"].split("/")[-1]))
                                visual_save_dir = os.path.dirname(visual_save_path)
                                if not os.path.isdir(visual_save_dir):
                                    os.makedirs(visual_save_dir)
                                try:
                                    visualization(os.path.join(image_folder,output["image_name"]), extracted, visual_save_path)
                                except Exception as e:
                                    print(e)
                                    print(output["query"])
                                    print(output["output"])
                                    print(extracted)
                            except Exception as e:
                                print(e)
                                print(output)
                                continue
                    elif args.model_type == 'griffon':
                        middle_brackets_pat = re.compile("(\[\d{1,}\.\d{3},\d{1,}\.\d{3},\d{1,}\.\d{3},\d{1,}\.\d{3}\])")
                        for j, output in enumerate(merged_outputs):
                            print(output)
                            try:
                                bbox = output["output"]
                                extracted = extract(bbox, middle_brackets_pat)
                                scores = output["scores"]
                                for idx, item in enumerate(extracted):
                                    try:
                                        predict_bbox = torch.tensor(xyxy2xywh(item["bbox"]), dtype=torch.float32).view(-1, 4)
                                        predict_bbox[:, ::2] *= output["width"]
                                        predict_bbox[:, 1::2] *= output["height"]
                                        predict_cate = item["category_name"]
                                        #
                                        if predict_cate.lower() == 'rabbit' or predict_cate.lower() == 'bunny':
                                            predict_cate = 'Cottontail_Rabbit'
                                        predict_id = catname2id[predict_cate.replace('_', '-')]
                                        ret = {
                                            "image_id": output["image_id"],
                                            "category_id": predict_id,
                                            "bbox": predict_bbox.tolist()[0],
                                            "score": scores[idx],
                                        }
                                        eval_list.append(ret)
                                    except Exception as e:
                                        print(e)
                                        print(output["query"])
                                        print(output["output"])
                                        print(extracted)
                                        continue
                                
                            except Exception as e:
                                print(e)
                                print(output)
                                continue

                    save_path = os.path.join(model_data_out_dir, "{}_eval_processed.json".format(ds))
                    save_file = open(save_path, "w")
                    json.dump(eval_list, save_file)
                    save_file.close()

                    cocoGt = COCO(dataset_path)
                    cocoDt = cocoGt.loadRes(save_path)
                    cocoeval = COCOeval(cocoGt, cocoDt, "bbox")
                    cocoeval.evaluate()
                    cocoeval.accumulate()
                    cocoeval.summarize()
        else:
            if torch.distributed.get_rank() == 0:
                # import pdb; pdb.set_trace()
                merged_outputs = torch.load(os.path.join(model_data_out_dir,"original_pred.pth"))
                print(f"Begin to eval {ds}")
                print("Begin to eval")

                eval_list = []
                for j, output in tqdm(enumerate(merged_outputs)):
                    try:
                        bbox = output["output"]
                        extracted = parse_json(bbox)
                        
                        try:                                                                                                                                                                         
                            extracted = ast.literal_eval(extracted)                                                                                                                                                           
                        except Exception as e:                                                                                           
                            end_idx = extracted.rfind('"}') + len('"}')                                                                                                                                           
                            truncated_text = extracted[:end_idx] + "]"                                               
                            extracted = ast.literal_eval(truncated_text)
                        for idx, item in enumerate(extracted):
                            #import pdb; pdb.set_trace()
                            try:
                                predict_bbox = torch.tensor(xyxy2xywh(item["bbox_2d"]), dtype=torch.float32).view(-1, 4)
                                predict_bbox[:, ::2] = predict_bbox[:, ::2] / output["input_width"] * output["width"]
                                predict_bbox[:, 1::2] = predict_bbox[:, 1::2] / output["input_height"] * output["height"]

                                predict_id = catname2id[item["label"].lower()]
                                ret = {
                                    "image_id": output["image_id"],
                                    "category_id": predict_id,
                                    "bbox": predict_bbox.tolist()[0],
                                    "score": 0.99,
                                }
                                eval_list.append(ret)
                            except:
                                continue
                        visual_save_path = os.path.join(model_data_out_dir, "detection/{}_{}".format(j,output["image_name"].split("/")[-1]))
                        visual_save_dir = os.path.dirname(visual_save_path)
                        if not os.path.isdir(visual_save_dir):
                            os.makedirs(visual_save_dir)
                        try:
                            _extracted = []
                            for extract_item in extracted:
                                ori_bbox = extract_item['bbox_2d']
                                resized_bbox = [ori_bbox[0]/output["input_width"], ori_bbox[1]/output["input_height"], ori_bbox[2]/output["input_width"], ori_bbox[3]/output["input_height"]]
                                _extract_item  = {'category_name': extract_item['label'], 'bbox': resized_bbox}
                                _extracted.append(_extract_item)
                        
                            visualization(os.path.join(image_folder,output["image_name"]), _extracted, visual_save_path)
                        except Exception as e:
                            print(e)
                            print(output["query"])
                            print(output["output"])
                            print(extracted)
                    except Exception as e:
                        print(e)
                        print(output["query"])
                        print(output["output"])
                        print(extracted)
                        continue
                save_path = os.path.join(model_data_out_dir, "{}_eval_processed.json".format(ds))
                save_file = open(save_path, "w")
                json.dump(eval_list, save_file)
                save_file.close()

                cocoGt = COCO(dataset_path)
                cocoDt = cocoGt.loadRes(save_path)
                cocoeval = COCOeval(cocoGt, cocoDt, "bbox")
                cocoeval.evaluate()
                cocoeval.accumulate()
                cocoeval.summarize()
        torch.distributed.barrier()