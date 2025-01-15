import random
import torch
import re

import numpy as np

from PIL import Image, ImageDraw, ImageFont
from typing import List, Union
from torchvision.ops.boxes import box_area

Box = List[Union[float, int]]
Boxes = List[Box]
BoxesSeq = List[Boxes]
DEFAULT_BOXES_PLACEHOLDER="<box>"
BOX_SPLIT_PLACEHOLDER="&"

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def box_xyxy_expand2square(box, *, w, h):
    # padding the top and bottom or left and right of the image
    if w == h:
        return box
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = x1, y1, x2, y2
        return box
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = x1, y1, x2, y2
    return box

def xywh2xyxy(bbox):
    # top left, wh
    x = np.asarray(bbox)
    y = np.zeros_like(x)
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = (x[:, 0] + x[:, 2])
    y[:, 3] = (x[:, 1] + x[:, 3])
    return y.tolist()

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

def xywh2cxcywh(bbox):
    # top left, wh 2 center x center y wh
    x = np.asarray(bbox)
    y = np.zeros_like(x)
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
    y[:, 0] = (x[:, 0] + 0.5 * x[:, 2])
    y[:, 1] = (x[:, 1] + 0.5 * x[:, 3])
    y[:, 2] = x[:, 2]
    y[:, 3] = x[:, 3]
    return y.tolist()

def xyxy2cxcywh(bbox):
    # top left bottom right 2 center x center y w h
    x = np.asarray(bbox)
    y = np.zeros_like(x)
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
    y[:, 0] = (x[:, 0] + x[:, 2])/2
    y[:, 1] = (x[:, 1] + x[:, 3])/2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y.tolist()

def cxcywh2xyxy(bbox):
    # center x center y w h 2 top left bottom right
    x = np.asarray(bbox)
    y = np.zeros_like(x)
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
    y[:, 0] = x[:, 0] - x[:, 2]/2
    y[:, 1] = x[:, 1] - x[:, 3]/2
    y[:, 2] = x[:, 0] + x[:, 2]/2
    y[:, 3] = x[:, 1] + x[:, 3]/2
    return y.tolist()

def xyxy_rotate(bbox, theta, shift_x=0, shift_y=0, scale=1):
    # 此处为逆时针旋转theta
    x = np.asarray(bbox)-np.array([0.5, 0.5, 0.5, 0.5])
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    theta = np.radians(theta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta],[sin_theta, cos_theta]])
    y = x.reshape(-1, 2).dot(rotation_matrix)
    y = y.reshape(-1, 4) + np.array([shift_x, shift_x, shift_x, shift_y])
    y = y / scale
    return y.tolist()

def xyxy_vers_rotate(bbox, theta, shift_x=0, shift_y=0, scale=1):
    # 此处为顺时针回旋转theta
    x = np.asarray(bbox)
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    theta = np.radians(theta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, sin_theta],[-sin_theta, cos_theta]])
    y = x.reshape(-1, 2).dot(rotation_matrix)
    y = y.reshape(-1, 4) + np.array([shift_x, shift_y, shift_x, shift_y])
    y = y / scale
    return y.tolist()

def norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box

    # Calculate the normalized coordinates with min-max clamping
    norm_x1 = max(0.0, min(x1 / w, 1.0))
    norm_y1 = max(0.0, min(y1 / h, 1.0))
    norm_x2 = max(0.0, min(x2 / w, 1.0))
    norm_y2 = max(0.0, min(y2 / h, 1.0))

    # Return the normalized box coordinates
    normalized_box = (round(norm_x1, 3), round(norm_y1, 3), round(norm_x2, 3), round(norm_y2, 3))
    return normalized_box

def de_norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box = x1, y1, x2, y2
    return box

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
    font = ImageFont.truetype("./Times New Roman.ttf", font_size)

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

def resize_box(box, size, height, width, pre=False):

    if "shortest_edge" in size:
        size = size["shortest_edge"]
    elif "height" in size:
        assert size["height"] == size["width"]
        size = size["width"]
        pre=True #set to true as it works like out resize
    else:
        raise AssertionError
    if not pre:

        min_size = size
        ratio = float(min_size) / min(height, width)
        x1, y1, x2, y2 = box
        return [x1 * ratio, y1 * ratio, x2* ratio, y2 * ratio], height*ratio, width*ratio
    else:
        square_size = size
        width_ratio = float(square_size) / width
        height_ratio = float(square_size) / height
        if len(box) >0 and isinstance(box[0], list):
            box = np.asarray(box)
            box[:, ::2] = box[:, ::2] * width_ratio
            box[:, 1::2] = box[:, 1::2] * height_ratio
            return box.tolist(), height*height_ratio, width*width_ratio
        elif len(box) == 4 and (isinstance(box[0], int) or isinstance(box[0], float)):
            x1, y1, x2, y2 = box
            #assert int(np.ceil(height*height_ratio)) == square_size and int(np.ceil(width*width_ratio)) == square_size, f"{height}, {height_ratio}, {width}, {width_ratio}, {square_size}"
            return [x1 * width_ratio, y1 * height_ratio, x2* width_ratio, y2 * height_ratio], height*height_ratio, width*width_ratio
        else:
            raise AssertionError

def center_crop_box(box, crop_size, height, width):
    x_minus = (width - crop_size["width"]) // 2
    y_minus = (height - crop_size["height"]) // 2
    x1, y1, x2, y2 = box
    x1 = min(max(x1 - x_minus, 0), crop_size["width"])
    y1 = min(max(y1 - y_minus, 0), crop_size["height"])
    x2 = min(max(x2 - x_minus, 0), crop_size["width"])
    y2 = min(max(y2 - y_minus, 0), crop_size["height"])
    return [x1, y1, x2, y2], crop_size["height"], crop_size["width"]

def merge_strings(strings):
    merged_dict = {}
    
    for string in strings:
        prefix, suffix = string.split('-', 1)
        if prefix in merged_dict:
            current_suffix = merged_dict[prefix]
            if len(suffix) > len(current_suffix):
                merged_dict[prefix] = suffix
        else:
            merged_dict[prefix] = suffix
    
    merged_strings = [prefix + '-' + suffix for prefix, suffix in merged_dict.items()]
    return merged_strings

def accum_probs(output_ids, output_scores, start_token=200348, end_token=50648):
    probs = []
    start_idx = 0
    end_idx = 0
    
    loc_start_idx = 0
    loc_end_idx = 0
    loc_probs = []

    output_ids = output_ids.squeeze(0)
    for idx, token in enumerate(output_ids):
        if token == start_token:
            start_idx = idx + 1
            loc_end_idx = idx -1
            temp = output_scores[loc_start_idx:loc_end_idx].max(dim=-1)[0]
            loc_prob = torch.prod(temp)
            loc_probs.append(loc_prob.item())
        elif token == end_token:
            end_idx = idx
            loc_start_idx = idx + 1
            prob = torch.prod(output_scores[start_idx:end_idx].max(dim=-1)[0])
            probs.append(prob.item())
        elif idx == len(output_ids)-1:
            loc_end_idx = idx
            loc_prob = torch.prod(output_scores[loc_start_idx:loc_end_idx].max(dim=-1)[0])
            loc_probs.append(loc_prob.item())

    if len(loc_probs) == len(probs):
        probs_out = torch.tensor(loc_probs) ** 0.5 * torch.tensor(probs)**0.5
    else:
        min_shape = min(len(loc_probs), len(probs))
        probs_out = torch.tensor(loc_probs[:min_shape]) ** 0.5 * torch.tensor(probs[:min_shape])**0.5
    return probs_out.numpy().tolist()