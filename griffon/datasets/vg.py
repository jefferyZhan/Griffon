import transformers
import os
import json
import torch
import copy
import cv2
import time

import numpy as np
from PIL import Image
from typing import Dict, Optional, Sequence, List
from torch.utils.data import Dataset
from torchvision import transforms

from .base_class import DataArguments, preprocess, preprocess_multimodal
from griffon.coor_utils import box_xyxy_expand2square, resize_box, visualization, TextBoxFormatter, BinBoxFormatter
from griffon.utils import auto_rank0_print

class VGDataset(Dataset):
    "dataset for vg task"
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 image_folder: str = None,
                 template_file: str = None,
                 prompt: str = None,
                 debug=False,
                 ):
        super(VGDataset, self).__init__()
        
        f = open(data_path, "r", encoding="utf-8")
        list_data_dict = f.readlines()

        auto_rank0_print("Formatting inputs...VG Dataset: jsonl_dir : {}  ,image_folder : {}".format(data_path,image_folder))
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_folder = image_folder
        assert not(prompt != None and template_file != None)
        if template_file is not None:
            self.prompts = json.load(open(template_file, 'r', encoding='utf8'))
            self.rng = np.random.default_rng(1203)
            self.prompt = None
        else:
            self.prompt = prompt # "<image><expr><single>"
            self.prompts = None

        if "bin" in data_args.formatter.lower():
            self.boxformat = BinBoxFormatter(self.data_args.image_processor, precision=3, box_pattern=self.data_args.box_pattern)
        else:
            self.boxformat = TextBoxFormatter(self.data_args.image_processor, precision=3, box_pattern=self.data_args.box_pattern)

        self.debug = debug
        if not self.data_args.image_processor.do_resize:
            self.resize = transforms.Resize((self.data_args.image_processor.size["shortest_edge"], self.data_args.image_processor.size["shortest_edge"]))

    def __len__(self):
        return len(self.list_data_dict)
    
    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128
            sample = json.loads(sample)
            if "bbox" in sample:
                box_num = len(sample["bbox"])
            else:
                box_num = len(sample["bboxes"])
            length_list.append(30 * box_num + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        # 在此处，纯文本的长度赋予-cur_len进行区分，同理，对于region，可进行该处理
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128
            sample = json.loads(sample)
            if "bbox" in sample:
                box_num = len(sample["bbox"])
            else:
                box_num = len(sample["bboxes"])
            length_list.append(30 * box_num + img_tokens)
        return length_list
    
    @property
    def task_lengths(self):
        # id = 2
        length_list = []
        for sample in self.list_data_dict:
            sample = json.loads(sample)
            if "bbox" in sample:
                box_num = len(sample["bbox"])
            else:
                box_num = len(sample["bboxes"])
            length_list.append((2, 30 * box_num))
        return length_list
    
    def transform2conv(self, i):
        if self.prompts is not None:
            prompt = self.rng.choice(self.prompts)
        else:
            prompt = self.prompt
        if not prompt.endswith("\n"):
            prompt += "\n"
        if "<image>" not in prompt:
            prompt = "<image>" + prompt
        # Add format output
        # prompt = prompt + "\nAnswer with the bounding box coordinates."
        # item为读取jsonl一行后获取的dict
        item = json.loads(self.list_data_dict[i])
        # get图像名称
        img_path = item["image_path"]
        # 指代描述
        expr = item["expression"]
        # 框坐标 x0 y0 x1 y1 
        bbox = item["bboxes"]
        # 长宽
        width = item["width"]
        height = item["height"]
        if self.data_args.attr:
            regions = item["discrete_ids"]
        
        if len(bbox) > 0:
            if self.data_args.image_aspect_ratio == 'pad':
                #需要对图片的padding进行处理
                #此处是在__get_item__之前将其处理为square之后，与之后同步
                bbox = box_xyxy_expand2square(bbox, width, height)
                width = max(width, height)
                height = max(width, height)
            if not self.data_args.image_processor.do_resize:
                bbox, height, width = resize_box(bbox, self.data_args.image_processor.size, height, width, pre=True)
            #将bbox处理为字符串
            sentence = "<box>"
            sentence = self.boxformat(sentence, [([expr]*len(bbox), bbox)], height, width) #这里由于处理函数的限制，需要满足BoxesSeq的需要
            if self.data_args.attr:
                instances = sentence.split("&")
                temp_instances = []
                if len(regions) != len(instances) or len(regions) == 0:
                    pass
                else:
                    for coor, region in zip(instances, regions):
                        region_ = [f"<region_{item:05d}>" for item in region]
                        region_s = "".join(region_)
                        new_s = coor + f"-<DIS>{region_s}</DIS>"
                        temp_instances.append(new_s)
                    sentence = "&".join(temp_instances)
        else:
            sentence = "None"
        ret = {
            "image": img_path,
            "conversations": [
                {
                    "from": "human",
                    "value": prompt.replace("<expr>", expr)
                },
                {
                    "from": "gpt",
                    "value": sentence
                }
            ]
        }
        if i == 0:
            print(prompt.replace("<expr>", expr))
            print(sentence)
        return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        #得到原始的item
        sources = self.transform2conv(i)
        # sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = sources[0]['image']
            if self.image_folder is None:
                image_folder = self.data_args.image_folder
            else:
                image_folder = self.image_folder
            processor = self.data_args.image_processor #CLIP中的process，将图像resize norm等
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            except:
                image = Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join(image_folder, image_file)), cv2.COLOR_BGR2RGB))
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean)) #预先padding成为square
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif not self.data_args.image_processor.do_resize:
                image = self.resize(image)
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            has_image = True
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            has_image = False
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=has_image)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if has_image:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        #Debug Mode. Check whether the normalization is right.
        if self.debug:
            bboxes = self.boxformat.extract(sources[0][-1]["value"])
            toPIL = transforms.ToPILImage()
            # visualization(toPIL(image), bboxes, f"../instrct_out/{image_file.split("/")[-1]}")
            if len(bboxes) > 0:
                visualization(toPIL(image), bboxes, "/public/home/zhuyousong/zhaohongyin/debug/rec{}".format(image_file.split("/")[-1]), self.data_args.box_pattern)
        if self.debug==True:

            try:
                bboxes = self.boxformat.extract(sources[0][-1]["value"])
                current_timestamp = time.time()
                local_time = time.localtime(current_timestamp)
                current_time = time.strftime("%Y-%m-%d_%H_%M_%S", local_time)
                visualization(os.path.join(image_folder, image_file), bboxes, "./debug/{}-{}.jpg".format(current_time, image_file.split("/")[-1]), self.data_args.box_pattern)
            except:
                print("Please improve the dataset to support this")
        return data_dict



