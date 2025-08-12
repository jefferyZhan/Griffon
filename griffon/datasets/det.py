from ast import Assert, expr_context
import transformers
import json
import torch
import copy
import os
import numpy as np
import random
import time

from torch.utils.data import Dataset
from pycocotools.coco import COCO
from typing import Dict, Optional, Sequence, List
from PIL import Image
from torchvision import transforms
import cv2

from .base_class import DataArguments, preprocess, preprocess_multimodal, preprocess_special_number
from griffon.coor_utils import TextBoxFormatter, BinBoxFormatter, box_xyxy_expand2square, xywh2xyxy, visualization, resize_box, reorder
from griffon.utils import auto_rank0_print

class DETDataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 image_folder: str = None,
                 prompt: str = None,
                 template_file: str = None,
                 debug=False,
                 cate_sample=False,):
        super(DETDataset, self).__init__()
        auto_rank0_print("Formatting inputs...DET Dataset")
        self.tokenizer = tokenizer
        #类名列表
        self.category_name_list=[]
        f = open(data_path, "r")
        lines = f.readlines()
        list_data_dict = []
        # Feature: Support training with removing instances over 10(N)
        # for line in lines:
        #     line = json.loads(line)
        #     length = 0
        #     for obj in line["objects"]:
        #         length += len(obj["bboxes"])
        #     if length <= 10:
        #         list_data_dict.append(line)
        # print("{}: removing item over 10 objects from {} to {}".format(data_path.split("/")[-1].split(".")[0], len(lines), len(list_data_dict)))
        for line in lines:
            line = json.loads(line)
            if len(line["objects"]) > 0:
                list_data_dict.append(line)
        self.list_data_dict = list_data_dict
        # Initialize class name list
        for ann in list_data_dict:
            try:
                # ann=json.loads(ann)
                rows=ann['objects']
            except:
                print(data_path)
                raise AssertionError
            if len(self.category_name_list)==80:
                print("DET cate_name_list init finished")
                break
            for row in rows:
                if row['category_name'] not in self.category_name_list:
                    self.category_name_list.append(row['category_name'])

        
        assert not (prompt != None and template_file != None)
        if template_file is not None:
            self.prompts = json.load(open(template_file, "r", encoding="utf8"))
            self.rng = np.random.default_rng(1203)
            self.prompt = None
        else:
            self.prompt = prompt
            self.prompts = None
        self.image_folder = image_folder

        # box representation type
        self.data_args = data_args
        if "bin" in data_args.formatter.lower():
            self.boxformat = BinBoxFormatter(self.data_args.image_processor, precision=3,box_pattern=self.data_args.box_pattern)
        else:
            self.boxformat = TextBoxFormatter(self.data_args.image_processor, precision=3,box_pattern=self.data_args.box_pattern)
        
        self.debug = debug
        self.cate_sample = cate_sample
        if not self.data_args.image_processor.do_resize:
            self.resize = transforms.Resize((self.data_args.image_processor.size["shortest_edge"], self.data_args.image_processor.size["shortest_edge"]))
    
    def __len__(self):
        return len(self.list_data_dict)

    def transform2conv(self, i):
        if self.prompts is not None:
            prompt = self.rng.choice(self.prompts)
        else:
            prompt = self.prompt
        # if not prompt.endswith("\n"):
        #     prompt += "\n"

        prompt = prompt.replace("The output format for each detected object is class name-[top-left coordinate,bottom-right coordinate], e.g. person-[0.001,0.345,0.111,0.678]. Concatenate them with &.", "")
        prompt += "\nAnswer in the format: class name-[x1, y1, x2, y2], and concatenate them with &."
        
        # item = json.loads(self.list_data_dict[i])
        item = self.list_data_dict[i]
        # image_path
        file_name = item["image_path"]

        # Process for Objects365
        if item["dataset_name"]=="OBJ365":
            file_name = "{}/{}".format(file_name.split("/")[-2],file_name.split("/")[-1])

        height = item["height"]
        width = item["width"]
        # init cate_names and bboxes 
        cate_names = []
        cate_bboxes = []  # build per image (class, corr-box)  bboxes format x0y0x1y1
        for ann in item['objects']:
            bboxes = ann["bboxes"]
            cate_name = ann["category_name"]
            for bbox in bboxes:
                cate_names.append(cate_name)
                cate_bboxes.append(bbox)
        
        # image transform
        if not self.data_args.image_processor.do_resize:
            cate_bboxes, _, _ = resize_box(cate_bboxes, self.data_args.image_processor.size, height, width, pre=True)
        if self.data_args.image_aspect_ratio == "pad":
            new_height = max(width, height)
            new_width = max(width, height)
        elif self.data_args.image_aspect_ratio == "org_pad":
            new_height = max(width, height)
            new_width = max(width, height)
        elif not self.data_args.image_processor.do_resize:
            new_height = self.data_args.image_processor.size["shortest_edge"]
            new_width = self.data_args.image_processor.size["shortest_edge"]
        else:
            new_height = height
            new_width = width

        # Random category indexes for each instance
        # Sample 80 from all categories with GT classes included 
        build_prompt_cates = list(set(cate_names))
        pool = list(set(self.category_name_list).difference(set(build_prompt_cates)))
        sampled = random.sample(pool, 80-len(build_prompt_cates))
        
        build_prompt_cates += sampled
        random.shuffle(build_prompt_cates)
        cate_expr = ", ".join(build_prompt_cates)
        prompt = prompt.replace("<category set>", cate_expr)
        #update self.category_name_list
        self.category_name_list=build_prompt_cates

        sentence = "<box>"
        #sequence bbox according to x1
        cate_names, cate_bboxes = reorder(cate_names, cate_bboxes)
        sentence = self.boxformat(sentence, [(cate_names, cate_bboxes)], new_height, new_width)
        ret = {
            "image": file_name,
            "conversations": [
                {
                    "from": "human",
                    "value": prompt
                },
                {
                    "from": "gpt",
                    "value": sentence
                }
            ]
        }
        if i == 0:
            print(prompt)
            print(sentence)
        return ret
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.transform2conv(i)
        # print(sources)
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
            processor = self.data_args.image_processor 
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            except:
                image = Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join(image_folder, image_file)), cv2.COLOR_BGR2RGB))
            if self.data_args.image_aspect_ratio == 'pad':# pass
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
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean)) 
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif self.data_args.image_aspect_ratio == "org_pad": #pass
                def expand2square_org(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, 0)) 
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, (0, 0))
                        return result
                image = expand2square_org(image, tuple(int(x*255) for x in processor.image_mean)) 
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif not self.data_args.image_processor.do_resize:
                image = self.resize(image)
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            has_image = True
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            has_image = False
        
        ### ZYF Modified for supporting number tokenizer ###
        # sources, number_recoder = preprocess_special_number(copy.deepcopy(sources))
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=has_image)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],)
                            #  numbers=number_recoder[0])

        # image exist in the data
        if has_image:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        #Debug Mode. Check whether the normalization is right.
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
    
    #add this property to ensure region only sample
    #each instance is set to 30
    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # sample = json.loads(sample)
            obj_length=0
            for obj in sample['objects']:
                obj_length+=len(obj['bboxes']) 
            img_tokens = 128
            length_list.append(30 * obj_length + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # sample = json.loads(sample)
            obj_length=0
            for obj in sample['objects']:
                obj_length+=len(obj['bboxes']) 
            img_tokens = 128
            length_list.append(30 * obj_length + img_tokens)
        return length_list

    @property
    def task_lengths(self):
        # id = 2
        length_list = []
        for sample in self.list_data_dict:
            # sample = json.loads(sample)
            obj_length=0
            for obj in sample['objects']:
                obj_length+=len(obj['bboxes']) 
            length_list.append((2, 30 * obj_length))
        return length_list

class RANDDETDataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 image_folder: str = None,
                 prompt: str = None,
                 template_file: str = None,
                 debug=False,
                 cate_sample=False,):
        super(RANDDETDataset, self).__init__()
        auto_rank0_print("Formatting inputs... RAND DET Dataset")
        self.tokenizer = tokenizer
        self.category_name_list=[]
        f = open(data_path, "r")
        lines = f.readlines()
       
        list_data_dict = []
        
        for line in lines:
            line = json.loads(line)
            if len(line["objects"]) > 0:
                list_data_dict.append(line)
        self.list_data_dict = list_data_dict
        
        assert not (prompt != None and template_file != None)
        if template_file is not None:
            self.prompts = json.load(open(template_file, "r", encoding="utf8"))
            self.rng = np.random.default_rng(1203)
            self.prompt = None
        else:
            self.prompt = prompt
            self.prompts = None
        self.image_folder = image_folder

        self.data_args = data_args
        if "bin" in data_args.formatter.lower():
            self.boxformat = BinBoxFormatter(self.data_args.image_processor, precision=3,box_pattern=self.data_args.box_pattern)
        else:
            self.boxformat = TextBoxFormatter(self.data_args.image_processor, precision=3,box_pattern=self.data_args.box_pattern)
        
        self.debug = debug
        self.cate_sample = cate_sample
        if not self.data_args.image_processor.do_resize:
            self.resize = transforms.Resize((self.data_args.image_processor.size["shortest_edge"], self.data_args.image_processor.size["shortest_edge"]))
    
    def __len__(self):
        return len(self.list_data_dict)

    def transform2conv(self, i):
        if self.prompts is not None:
            prompt = self.rng.choice(self.prompts)
        else:
            prompt = self.prompt
        if not prompt.endswith("\n"):
            prompt += "\n"
        if "<image>" not in prompt:
            prompt = "<image>" + prompt
        
        # item = json.loads(self.list_data_dict[i])
        item = self.list_data_dict[i]
        # image_path
        file_name = item["image_path"]

        # Process for Objects365
        if item["dataset_name"]=="OBJ365":
            file_name = "{}/{}".format(file_name.split("/")[-2],file_name.split("/")[-1])

        height = item["height"]
        width = item["width"]
        cate_names = []
        cate_bboxes = []  # build per image (class, corr-box)  bboxes format x0y0x1y1
        neg_cate_names = []
        for ann in item['objects']:
            bboxes = ann["bboxes"]
            cate_name = ann["category_name"]
            if len(bboxes) > 0:
                for bbox in bboxes:
                    cate_names.append(cate_name)
                    cate_bboxes.append(bbox)
            else:
                neg_cate_names.append(cate_name)
        
        # 图像变换
        if not self.data_args.image_processor.do_resize:
            cate_bboxes, _, _ = resize_box(cate_bboxes, self.data_args.image_processor.size, height, width, pre=True)
        if self.data_args.image_aspect_ratio == "pad":
            new_height = max(width, height)
            new_width = max(width, height)
        elif self.data_args.image_aspect_ratio == "org_pad":
            new_height = max(width, height)
            new_width = max(width, height)
        elif not self.data_args.image_processor.do_resize:
            # 直接resize ！！！
            new_height = self.data_args.image_processor.size["shortest_edge"]
            new_width = self.data_args.image_processor.size["shortest_edge"]
        else:
            new_height = height
            new_width = width

        build_prompt_cates = list(set(cate_names + neg_cate_names))
        
        random.shuffle(build_prompt_cates)
        if len(build_prompt_cates) == 1:
            cate_expr = build_prompt_cates[0]
        else:
            cate_expr = ", ".join(build_prompt_cates[:-1])
            if len(build_prompt_cates) == 2:
                cate_expr += " and {}".format(build_prompt_cates[-1])
            else:
                cate_expr += ", and {}".format(build_prompt_cates[-1])
        prompt = prompt.replace("<expr>", cate_expr)
        self.category_name_list=build_prompt_cates

        sentence = "<box>"
        cate_names, cate_bboxes = reorder(cate_names, cate_bboxes)
        sentence = self.boxformat(sentence, [(cate_names, cate_bboxes)], new_height, new_width)
        ret = {
            "image": file_name,
            "conversations": [
                {
                    "from": "human",
                    "value": prompt
                },
                {
                    "from": "gpt",
                    "value": sentence
                }
            ]
        }
        if i == 0:
            print(prompt)
            print(sentence)
        return ret
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        sources = self.transform2conv(i)
        # print(sources)
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
            if self.data_args.image_aspect_ratio == 'pad':# pass
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
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif self.data_args.image_aspect_ratio == "org_pad": #pass
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
                image = expand2square_org(image, tuple(int(x*255) for x in processor.image_mean)) #预先padding成为square
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif not self.data_args.image_processor.do_resize:
                image = self.resize(image)
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            has_image = True
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            has_image = False
        
        ### ZYF Modified for supporting number tokenizer ###
        # sources, number_recoder = preprocess_special_number(copy.deepcopy(sources))
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=has_image)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],)
                            #  numbers=number_recoder[0])

        # image exist in the data
        if has_image:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        #Debug Mode. Check whether the normalization is right.
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
    
    #add this property to ensure region only sample
    #each instance is set to 30
    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # sample = json.loads(sample)
            obj_length=0
            for obj in sample['objects']:
                obj_length+=len(obj['bboxes']) 
            img_tokens = 128
            length_list.append(30 * obj_length + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # sample = json.loads(sample)
            obj_length=0
            for obj in sample['objects']:
                obj_length+=len(obj['bboxes']) 
            img_tokens = 128
            length_list.append(30 * obj_length + img_tokens)
        return length_list

    @property
    def task_lengths(self):
        # id = 2
        length_list = []
        for sample in self.list_data_dict:
            # sample = json.loads(sample)
            obj_length=0
            for obj in sample['objects']:
                obj_length+=len(obj['bboxes']) 
            length_list.append((2, 30 * obj_length))
        return length_list