import datetime
import logging
import logging.handlers
import os
import sys
import re
import requests
import numpy as np
from PIL import ImageDraw, Image, ImageFont
import torch
import random

from llava.constants import LOGDIR, BOX_SPLIT_PLACEHOLDER
from llava.datasets import smart_tokenizer_and_embedding_resize

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"

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

def visualization(image_path, extract_bboxes, save_path):
    if isinstance(image_path, str):
        image = Image.open(image_path)
        # image = image.resize((2*width, 2*height))
        draw = ImageDraw.Draw(image)
    else:
        image = image_path
        # image = image.resize((2*width, 2*height))
        draw = ImageDraw.Draw(image)
    height = image.height
    width = image.width


    line_width = int(width * 0.005) if width * 0.005 > 2 else 2
    font_size = int(height * 0.025) if height * 0.025 > 15 else 15 
    font = ImageFont.truetype("./Times New Roman.ttf", font_size)

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

    for i, bbox in enumerate(bboxes):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        draw.rectangle(bbox, outline=color, width=line_width)

        text = classes[i]
        text_w, text_h = draw.textsize(text, font=font)
        draw.rectangle([bbox[0], bbox[1]-text_h, bbox[0]+text_w, bbox[1]], fill=color)
        draw.text((bbox[0], bbox[1]-text_h), text, fill=(255, 255, 255), font=font)
    
    image.save(save_path)

from typing import List, Union
from llava.constants import DEFAULT_BOXES_PLACEHOLDER
Box = List[Union[float, int]]
Boxes = List[Box]
BoxesSeq = List[Boxes]

def resize_box(box, size, height, width, pre=False):
    # if pre is set to true, pre resize
    if not pre:
        #single box
        assert "shortest_edge" in size
        min_size = size["shortest_edge"]
        ratio = float(min_size) / min(height, width)
        x1, y1, x2, y2 = box
        return [x1 * ratio, y1 * ratio, x2* ratio, y2 * ratio], height*ratio, width*ratio
    else:
        square_size = size["shortest_edge"]
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

class BoxFormatter:
    def __init__(self, preprocessor, bboxes_token=DEFAULT_BOXES_PLACEHOLDER):
        self.bboxes_token = bboxes_token
        # normally the bboxes_token_pat is the same as bboxes_token if u not use some weird token
        self.bboxes_token_pat = re.compile(bboxes_token)
        self.preprocessor = preprocessor
        self.do_resize = self.preprocessor.do_resize
        if self.do_resize:
            self.size = self.preprocessor.size
        else:
            self.size = None
        self.do_center_crop = self.preprocessor.do_center_crop
        if self.do_center_crop:
            self.crop_size = self.preprocessor.crop_size
        else:
            self.crop_size = None

    def __call__(self, sentence: str, bboxes_seq, height, width) -> str:
        all_box = self.bboxes_token_pat.findall(sentence)
        assert len(all_box) == len(bboxes_seq), f"not match. sentence: {sentence}. boxes:{bboxes_seq}"
        if len(all_box) == 0:
            return sentence
        bboxes_strs = [self.format_box(self.bbox_processor(height, width, bboxes)) for bboxes in bboxes_seq]
        converted = sentence.replace(self.bboxes_token, '{}').format(*bboxes_strs)
        return converted

    def format_box(self, bboxes: Boxes) -> str:
        raise NotImplementedError

    def extract(self, string: str) -> List[Boxes]:
        raise NotImplementedError
    
    def bbox_processor(self, org_height, org_width, bboxes):
        new_bboxes = []
        for box in bboxes:
            # for a new box, the height and width should be reset to original one
            if self.do_resize:
                box, height, width = resize_box(box, self.size, org_height, org_width)
            else:
                box = box
                height = org_height
                width = org_width
            if self.do_center_crop:
                box, height, width = center_crop_box(box, self.crop_size, height, width)
            else:
                box = box
                height = height
                width = width
            box = norm_box_xyxy(box, w=width, h=height)
            new_bboxes.append(box)
        return new_bboxes

class NumberBoxFormatter(BoxFormatter):
    def __init__(self, *args, precision=3, use_small_brackets=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = precision #precision代表着小数点后几位
        self.use_small_brackets = use_small_brackets

        small_brackets_pat = re.compile(r'\(\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\)')

        middle_brackets_pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\]')

        self.pat = small_brackets_pat if use_small_brackets else middle_brackets_pat

    def format_box(self, boxes: Boxes) -> str:
        """
            Input: [[0.123, 0.22, 1.000, 0.998],[0.123, 0.22, 1.000, 0.998]] # normalized
            Output: "[0.123,0.220,1.000,0.998;0.123, 0.22, 1.000, 0.998]" #之后的";"可能需要处理
        """
        box_strs = []
        for box in boxes:
            box_str = ','.join([f"{elem:.{self.precision}f}" for elem in box])
            if self.use_small_brackets:
                box_str = "(" + box_str + ")"
            else:
                box_str = "[" + box_str + "]"
            box_strs.append(box_str)
        boxes_str = '&'.join(box_strs)
        return boxes_str

    def extract(self, string: str) -> List[Boxes]:
        """ balabala<boxes>balabala<boxes> -> [boxes, boxes] """
        ret = []
        for bboxes_str in self.pat.findall(string):
            bboxes = []
            bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
            for bbox_str in bbox_strs:
                bbox = list(map(float, bbox_str.split(',')))
                bboxes.append(bbox)
            ret.append(bboxes)
        return ret

class MultiNumberBoxFormatter(BoxFormatter):
    def __init__(self, *args, precision=3, use_small_brackets=False, box_split_placeholder=BOX_SPLIT_PLACEHOLDER, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.precision = precision
        self.use_small_brackets = use_small_brackets
        middle_brackets_pat_single = re.compile(r"\w+-\[\d\.\d{3},\d\.\d{3},\d\.\d{3},\d\.\d{3}\]")
        middle_brackets_pat_multi = re.compile(r"\w+ \w+-\[\d\.\d{3},\d\.\d{3},\d\.\d{3},\d\.\d{3}\]")

        small_brackets_pat_single = re.compile(r"\w+-\(\d\.\d{3},\d\.\d{3},\d\.\d{3},\d\.\d{3}\)")
        small_brackets_pat_multi = re.compile(r"\w+ \w+-\(\d\.\d{3},\d\.\d{3},\d\.\d{3},\d\.\d{3}\)")
        self.pat = (small_brackets_pat_single, small_brackets_pat_multi) if use_small_brackets else (middle_brackets_pat_single, middle_brackets_pat_multi)
        self.box_split_placeholder = box_split_placeholder

    def __call__(self, sentence: str, bboxes_seq, height, width) -> str:
        # bboxes_seq: [([cls,...],[[x1,y1, x2, y2],...]), ...]
        # each item in bboxes_seq means the whole cls-box paris in one image
        # the number of <boxes> must match the number of bboxes
        all_box = self.bboxes_token_pat.findall(sentence)
        assert len(all_box) == len(bboxes_seq), f"not match. sentence: {sentence}. boxes:{bboxes_seq}"
        if len(all_box) == 0:
            return sentence
        bboxes_strs = [self.format_box(clses, self.bbox_processor(height, width, bboxes)) for clses, bboxes in bboxes_seq]
        converted = sentence.replace(self.bboxes_token, '{}').format(*bboxes_strs)
        return converted
    
    def format_box(self, clses, bboxes: Boxes) -> str:
        per_box_seq = []
        for cls, box in zip(clses, bboxes):
            # Remove edge box
            if (box[0] == 0 and box[2] == 0) or (box[0] == 1 and box[2] == 1):
                continue
            elif (box[1] == 0 and box[3] == 0) or (box[1] == 1 and box[3] == 1):
                continue
            box_seq = ','.join([f"{elem:.{self.precision}f}" for elem in box])
            if self.use_small_brackets:
                box_seq = "(" + box_seq + ")"
            else:
                box_seq = "[" + box_seq + "]"
            full_seq = cls + "-" + box_seq 
            per_box_seq.append(full_seq)
        if len(per_box_seq) == 0:
            return ['None']
        else:
            return self.box_split_placeholder.join(per_box_seq)
        # return self.box_split_placeholder.join(per_box_seq)

    def extract(self, string: str):
        # input: string above
        # output: [{"category_name": cls, "bbox": bbox}]
        output = []
        #box_strings = string.strip().split(self.box_split_placeholder)
        box_strings = self.pat[0].findall(string) + self.pat[1].findall(string)
        for b_str in box_strings:
            cls, bbox_str  = b_str.split("-")
            bbox_str = bbox_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
            bbox = [item for item in map(float, bbox_str.split(','))]
            ins = {
                "category_name": cls,
                "bbox": bbox
            }
            output.append(ins)

        return output

