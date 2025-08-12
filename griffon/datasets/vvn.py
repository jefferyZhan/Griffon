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
from PIL import Image, ImageOps
from torchvision import transforms
import cv2
from transformers import CLIPImageProcessor

from .base_class import DataArguments, preprocess, preprocess_multimodal
from griffon.coor_utils import TextBoxFormatter, BinBoxFormatter, box_xyxy_expand2square, xywh2xyxy, visualization, resize_box, reorder


class VisualDetDataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 image_folder: str = None,
                 prompt: str = None,
                 template_file: str = None,
                 unshared_visual_encoder: bool = False,
                 debug=False):
        super(VisualDetDataset, self).__init__()
        print("Formatting inputs... Visual Det Dataset")
        self.tokenizer = tokenizer
        f = open(data_path, "r")
        whole_annotations = json.load(f)
        list_images = whole_annotations["images"]
        catid2name = {cate["id"]: cate["name"] for cate in whole_annotations["categories"]}
        
        assert not (prompt != None and template_file != None)
        if template_file is not None:
            self.prompts = json.load(open(template_file, "r", encoding="utf8"))
            self.rng = np.random.default_rng(1203)
            self.prompt = None
        else:
            self.prompt = prompt
            self.prompts = None
        if len(catid2name) > 0:
            # fscd_lvis no cate
            self.catid2name = catid2name
        else:
            self.catid2name = None
        self.image_folder = image_folder
        self.api = COCO(data_path)
        if "instances" in data_path and os.path.exists(data_path.replace("instances","count")):
            #if use fscd_lvis, the examples are in the count file.
            exm_path = data_path.replace("instances","count")
            self.exm_api = COCO(exm_path)
        else:
            self.exm_api = None
        #filter images with no bbox
        # list_images = [image for image in list_images if (len(self.api.getAnnIds([image["id"]]))>0)]#增加要求，要求所有的ann的数量少于100
        # print("remove {} images that contain anns <0".format(len(whole_annotations["images"]) - len(list_images)))
        list_images = [image for image in list_images if (len(self.api.getAnnIds([image["id"]]))>0)]#增加要求，要求所有的ann的数量少于100
        print("remove {} images that contain anns <0".format(len(whole_annotations["images"]) - len(list_images)))
        self.list_images = list_images
        
        self.data_args = data_args
        if "bin" in data_args.formatter.lower():
            self.boxformat = BinBoxFormatter(self.data_args.image_processor, precision=3, box_pattern=self.data_args.box_pattern)
        else:
            self.boxformat = TextBoxFormatter(self.data_args.image_processor, precision=3, box_pattern=self.data_args.box_pattern)
        if unshared_visual_encoder:
            self.prompt_processor = CLIPImageProcessor.from_pretrained("checkpoints/clip-vit-large-patch14")
            self.prompt_resize = transforms.Resize((self.prompt_processor.size["shortest_edge"], self.prompt_processor.size["shortest_edge"]))
        else:
            self.prompt_processor = None
            self.prompt_resize = None
        self.debug = debug
        if not self.data_args.image_processor.do_resize:
            self.resize = transforms.Resize((self.data_args.image_processor.size["shortest_edge"], self.data_args.image_processor.size["shortest_edge"]))
    
    def __len__(self):
        return len(self.list_images)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_images:
            img_tokens = 128
            length_list.append(30 * len(self.api.getAnnIds(sample["id"])) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        # 在此处，纯文本的长度赋予-cur_len进行区分，同理，对于region，可进行该处理
        length_list = []
        for sample in self.list_images:
            length_list.append(-30 * len(self.api.getAnnIds(sample["id"])))
        return length_list

    def transform2conv(self, i):
        if self.prompts is not None:
            prompt = self.rng.choice(self.prompts)
        else:
            prompt = self.prompt
        if not prompt.endswith("\n"):
            prompt += "\n"
        
        image_dict = self.list_images[i]
        image_id = image_dict["id"]
        file_name = image_dict["file_name"]
        # Process for Objects365
        if "patch" in file_name:
            file_name = "{}/{}".format(file_name.split("/")[-2],file_name.split("/")[-1])
        height = image_dict["height"]
        width = image_dict["width"]
        ann_ids = self.api.getAnnIds([image_id])
        anns = self.api.loadAnns(ann_ids)
        cate_names = []
        cate_ids = []
        bboxes = []  # build per image (class, corr-box)
        for ann in anns:
            bbox = ann["bbox"]
            cate_id_box = ann["category_id"]
            cate_ids.append(cate_id_box)
            if self.catid2name is not None:
                cate_name_box = self.catid2name[cate_id_box]
            else:
                cate_name_box = "target"
            cate_names.append(cate_name_box)
            if self.data_args.image_aspect_ratio == "pad":
                bbox = box_xyxy_expand2square(bbox, width, height)
            bboxes.append(bbox)
        
        #There random sampling a category and record the temp.
        if self.exm_api is None:
            # if there is no exm file, randomly choose from the set
            chosen_category = random.choice(cate_ids)
            chosen_category_name = self.catid2name[chosen_category]
            index_chosen = torch.where(torch.tensor(cate_ids) == chosen_category)
            bboxes = (torch.tensor(bboxes)[index_chosen]).numpy().tolist()
            chosen_bbox = random.choice(bboxes) # still in x,y,w,h
        else:
            assert len(list(set(cate_names))) == 1, "all the boxes loaded should be the same category."
            exm_ann = self.exm_api.getAnnIds([image_id])
            exm_bbox = self.exm_api.loadAnns(exm_ann)[0]["boxes"] # x,y,w,h
            chosen_bbox = random.choice(exm_bbox)
            chosen_category_name = "target"

        # COCO:(x,y,w,h) turn to xyxy
        bboxes = xywh2xyxy(bboxes)
        if not self.data_args.image_processor.do_resize:
            bboxes, _, _ = resize_box(bboxes, self.data_args.image_processor.size, height, width, pre=True)
        if self.data_args.image_aspect_ratio == "pad":
            new_height = max(width, height)
            new_width = max(width, height)
        elif self.data_args.image_aspect_ratio == "org_pad":
            # 直接padding右侧 或者下方，从而保持原始的横纵比，且坐标直接resize即可，不需要进行padding处理
            new_height = max(width, height)
            new_width = max(width, height)
        elif not self.data_args.image_processor.do_resize:
            # 直接resize ！！！
            new_height = self.data_args.image_processor.size["shortest_edge"]
            new_width = self.data_args.image_processor.size["shortest_edge"]
        else:
            new_height = height
            new_width = width

        sentence = "<box>"
        #对bbox的顺序进行纠正,按照x1从大到小的顺序
        cate_names, bboxes = reorder(len(bboxes)*[chosen_category_name], bboxes)
        sentence = self.boxformat(sentence, [(len(bboxes)*[chosen_category_name], bboxes)], new_height, new_width)
        #去掉其中的类别
        sentence = sentence.replace(chosen_category_name, "").replace("-", "")
        ret = {
            "image": file_name,
            "region_bbox": chosen_bbox,
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
            image = ImageOps.exif_transpose(image)

            # crop the region out of the bbox 
            region_bbox = sources[0]["region_bbox"]
            region = image.crop((region_bbox[0], region_bbox[1], region_bbox[0]+region_bbox[2], region_bbox[1]+region_bbox[3]))

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
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif self.data_args.image_aspect_ratio == "org_pad":
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
                if self.prompt_resize is None:
                    region = self.resize(region)
                else:
                    region = self.prompt_resize(region)
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                    if self.prompt_resize is None:
                        region = processor.preprocess(region, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                    else:
                        region = self.prompt_processor.preprocess(region, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    if self.prompt_resize is None:
                        region = processor.preprocess(region, return_tensors='pt')['pixel_values'][0]
                    else:
                        region = self.prompt_processor.preprocess(region, return_tensors='pt')['pixel_values'][0]
            else:
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            has_image = True
            has_region = True
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            has_image = False
            has_region = False
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
        
        if has_region:
            data_dict["region"] = region
        
        #Debug Mode. Check whether the normalization is right.
        if self.debug or (i in range(10)):
            toPIL = transforms.ToPILImage()
            region = toPIL(region)
            region.save("debug/{}-region.jpg".format(image_file.split("/")[-1]))
            bboxes = self.boxformat.extract(sources[0][-1]["value"])
            current_timestamp = time.time()
            local_time = time.localtime(current_timestamp)
            current_time = time.strftime("%Y-%m-%d_%H_%M_%S", local_time)
            try:
                visualization(os.path.join(image_folder, image_file), bboxes, "debug/{}-{}.jpg".format(current_time, image_file.split("/")[-1]), self.data_args.box_pattern)
            except:
                print(sources[0])
        return data_dict
    
class VisualDetDataset_JSONL(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 image_folder: str = None,
                 prompt: str = None,
                 template_file: str = None,
                 unshared_visual_encoder: bool = False,
                 debug=False):
        super(VisualDetDataset_JSONL, self).__init__()
        print("Formatting inputs... Visual Det Dataset from jsonl file")
        self.tokenizer = tokenizer
        f = open(data_path, "r")
        list_images = f.readlines()
        self.list_images = list_images
        
        assert not (prompt != None and template_file != None)
        if template_file is not None:
            self.prompts = json.load(open(template_file, "r", encoding="utf8"))
            self.rng = np.random.default_rng(1203)
            self.prompt = None
        else:
            self.prompt = prompt
            self.prompts = None

        self.image_folder = image_folder

        if "instances" in data_path and os.path.exists(data_path.replace("instances","count")):
            #if use fscd_lvis, the examples are in the count file.
            exm_path = data_path.replace("instances","count")
            self.exm_api = COCO(exm_path)
        else:
            self.exm_api = None
        
        self.data_args = data_args
        if "bin" in data_args.formatter.lower():
            self.boxformat = BinBoxFormatter(self.data_args.image_processor, precision=3, box_pattern=self.data_args.box_pattern)
        else:
            self.boxformat = TextBoxFormatter(self.data_args.image_processor, precision=3, box_pattern=self.data_args.box_pattern)
        if unshared_visual_encoder:
            self.prompt_processor = CLIPImageProcessor.from_pretrained("checkpoints/clip-vit-large-patch14")
            self.prompt_resize = transforms.Resize((self.prompt_processor.size["shortest_edge"], self.prompt_processor.size["shortest_edge"]))
        else:
            self.prompt_processor = None
            self.prompt_resize = None
        self.debug = debug
        if not self.data_args.image_processor.do_resize:
            self.resize = transforms.Resize((self.data_args.image_processor.size["shortest_edge"], self.data_args.image_processor.size["shortest_edge"]))
    
    def __len__(self):
        return len(self.list_images)
    
    @property
    def lengths(self):
        length_list = []
        for sample in self.list_images:
            img_tokens = 128
            sample = json.loads(sample)
            box_num = len(sample["box"])
            length_list.append(30 * box_num + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        # 在此处，纯文本的长度赋予-cur_len进行区分，同理，对于region，可进行该处理
        length_list = []
        for sample in self.list_images:
            sample = json.loads(sample)
            box_num = len(sample["box"])
            length_list.append(-30 * box_num)
        return length_list

    def transform2conv(self, i):
        if self.prompts is not None:
            prompt = self.rng.choice(self.prompts)
        else:
            prompt = self.prompt
        if not prompt.endswith("\n"):
            prompt += "\n"
        
        image_dict = json.loads(self.list_images[i])
        file_name = image_dict["img_path"]
        height = image_dict["height"]
        width = image_dict["width"]

        category2boxes = {}
        for category, bbox in zip(image_dict["category_id"], image_dict["box"]):
            if self.data_args.image_aspect_ratio == "pad":
                bbox = box_xyxy_expand2square(bbox, width, height)
            if category in category2boxes:
                category2boxes[category].append(bbox)
            else:
                category2boxes[category] = [bbox] #box in xyxy
        
        #There random sampling a category and record the temp.

        # randomly choose from the set
        chosen_category = random.choice(list(category2boxes.keys()))
        bboxes = category2boxes[chosen_category]
        chosen_bbox = random.choice(bboxes) # still in x,y,w,h

        # jsonl xyxy
        if not self.data_args.image_processor.do_resize:
            bboxes, _, _ = resize_box(bboxes, self.data_args.image_processor.size, height, width, pre=True)
        if self.data_args.image_aspect_ratio == "pad":
            new_height = max(width, height)
            new_width = max(width, height)
        elif self.data_args.image_aspect_ratio == "org_pad":
            # 直接padding右侧 或者下方，从而保持原始的横纵比，且坐标直接resize即可，不需要进行padding处理
            new_height = max(width, height)
            new_width = max(width, height)
        elif not self.data_args.image_processor.do_resize:
            # 直接resize ！！！
            new_height = self.data_args.image_processor.size["shortest_edge"]
            new_width = self.data_args.image_processor.size["shortest_edge"]
        else:
            new_height = height
            new_width = width

        sentence = "<box>"
        #对bbox的顺序进行纠正,按照x1从大到小的顺序
        cate_names, bboxes = reorder(len(bboxes)*[chosen_category], bboxes)
        sentence = self.boxformat(sentence, [(len(bboxes)*[chosen_category], bboxes)], new_height, new_width)
        #去掉其中的类别
        sentence = sentence.replace(chosen_category, "").replace("-", "")
        ret = {
            "image": file_name,
            "region_bbox": chosen_bbox,
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

            # crop the region out of the bbox 
            region_bbox = sources[0]["region_bbox"]
            region = image.crop((region_bbox[0], region_bbox[1], region_bbox[0]+region_bbox[2], region_bbox[1]+region_bbox[3]))

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
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif self.data_args.image_aspect_ratio == "org_pad":
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
                if self.prompt_resize is None:
                    region = self.resize(region)
                else:
                    region = self.prompt_resize(region)
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                    if self.prompt_resize is None:
                        region = processor.preprocess(region, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                    else:
                        region = self.prompt_processor.preprocess(region, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    if self.prompt_resize is None:
                        region = processor.preprocess(region, return_tensors='pt')['pixel_values'][0]
                    else:
                        region = self.prompt_processor.preprocess(region, return_tensors='pt')['pixel_values'][0]
            else:
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            has_image = True
            has_region = True
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            has_image = False
            has_region = False
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
        
        if has_region:
            data_dict["region"] = region
        
        #Debug Mode. Check whether the normalization is right.
        if self.debug or (i in range(10)):
            toPIL = transforms.ToPILImage()
            region = toPIL(region)
            region.save("debug/{}-region.jpg".format(image_file.split("/")[-1]))
            bboxes = self.boxformat.extract(sources[0][-1]["value"])
            current_timestamp = time.time()
            local_time = time.localtime(current_timestamp)
            current_time = time.strftime("%Y-%m-%d_%H_%M_%S", local_time)
            try:
                visualization(os.path.join(image_folder, image_file), bboxes, "debug/{}-{}.jpg".format(current_time, image_file.split("/")[-1]), self.data_args.box_pattern)
            except:
                pass
        return data_dict
    
class VisualDetDataset_JSONL_v2(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 image_folder: str = None,
                 prompt: str = None,
                 template_file: str = None,
                 unshared_visual_encoder: bool = False,
                 debug=False):
        super(VisualDetDataset_JSONL_v2, self).__init__()
        print("Formatting inputs... Visual Det Dataset from jsonl file")
        self.tokenizer = tokenizer
        f = open(data_path, "r")
        list_images = f.readlines()
        self.list_images = list_images
        
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
            self.boxformat = BinBoxFormatter(self.data_args.image_processor, precision=3, box_pattern=self.data_args.box_pattern)
        else:
            self.boxformat = TextBoxFormatter(self.data_args.image_processor, precision=3, box_pattern=self.data_args.box_pattern)
        if unshared_visual_encoder:
            self.prompt_processor = CLIPImageProcessor.from_pretrained("checkpoints/clip-vit-large-patch14")
            self.prompt_resize = transforms.Resize((self.prompt_processor.size["shortest_edge"], self.prompt_processor.size["shortest_edge"]))
        else:
            self.prompt_processor = None
            self.prompt_resize = None
        self.debug = debug
        if not self.data_args.image_processor.do_resize:
            self.resize = transforms.Resize((self.data_args.image_processor.size["shortest_edge"], self.data_args.image_processor.size["shortest_edge"]))
    
    def __len__(self):
        return len(self.list_images)
    
    @property
    def lengths(self):
        length_list = []
        for sample in self.list_images:
            img_tokens = 128
            sample = json.loads(sample)
            obj_length=0
            for obj in sample['objects']:
                obj_length+=len(obj['bboxes']) 
            length_list.append(30 * obj_length + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        # 在此处，纯文本的长度赋予-cur_len进行区分，同理，对于region，可进行该处理
        length_list = []
        for sample in self.list_images:
            sample = json.loads(sample)
            obj_length=0
            for obj in sample['objects']:
                obj_length+=len(obj['bboxes']) 
            length_list.append(-30 * obj_length)
        return length_list

    @property
    def task_lengths(self):
        # id = 3
        length_list = []
        for sample in self.list_images:
            sample = json.loads(sample)
            obj_length=0
            for obj in sample['objects']:
                obj_length+=len(obj['bboxes']) 
            length_list.append((3, 30 * obj_length))
        return length_list

    def transform2conv(self, i):
        if self.prompts is not None:
            prompt = self.rng.choice(self.prompts)
        else:
            prompt = self.prompt
        
        # if not prompt.endswith("\n"):
        #     prompt += "\n"
        prompt += "\nAnswer in the format: [x1, y1, x2, y2], and concatenate them with &."
        
        image_dict = json.loads(self.list_images[i])
        file_name = image_dict["image_path"]
        height = image_dict["height"]
        width = image_dict["width"]

        index = random.choice(range(len(image_dict["objects"])))
        raw_bboxes = image_dict["objects"][index]["bboxes"] #xyxy
        if self.data_args.image_aspect_ratio == "pad":
            bboxes = []
            for bbox in raw_bboxes:
                bbox = box_xyxy_expand2square(bbox, width, height)
                bboxes.append(bbox)
        else:
            bboxes = copy.deepcopy(raw_bboxes)
        category_name = image_dict["objects"][index]["category_name"]
        
        #There random sampling a category and record the temp.
        # randomly choose from the set
        
        chosen_bbox = random.choice(bboxes) # Here in xyxy

        # jsonl xyxy
        if not self.data_args.image_processor.do_resize:
            bboxes, _, _ = resize_box(bboxes, self.data_args.image_processor.size, height, width, pre=True)
        if self.data_args.image_aspect_ratio == "pad":
            new_height = max(width, height)
            new_width = max(width, height)
        elif self.data_args.image_aspect_ratio == "org_pad":
            # 直接padding右侧 或者下方，从而保持原始的横纵比，且坐标直接resize即可，不需要进行padding处理
            new_height = max(width, height)
            new_width = max(width, height)
        elif not self.data_args.image_processor.do_resize:
            # 直接resize ！！！
            new_height = self.data_args.image_processor.size["shortest_edge"]
            new_width = self.data_args.image_processor.size["shortest_edge"]
        else:
            new_height = height
            new_width = width

        sentence = "<box>"
        #对bbox的顺序进行纠正,按照x1从大到小的顺序
        cate_names, bboxes = reorder(len(bboxes)*[category_name], bboxes)
        sentence = self.boxformat(sentence, [(len(bboxes)*[category_name], bboxes)], new_height, new_width)
        #去掉其中的类别
        sentence = sentence.replace(category_name, "").replace("-", "")
        ret = {
            "image": file_name,
            "region_bbox": chosen_bbox,
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
        #得到原始的item
        sources = self.transform2conv(i)
        if self.debug:
            print(sources)
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

            # crop the region out of the bbox 
            region_bbox = sources[0]["region_bbox"]
            region = image.crop((region_bbox[0], region_bbox[1], region_bbox[2], region_bbox[3]))
            if self.debug:
                region.save("./region.png")

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
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif self.data_args.image_aspect_ratio == "org_pad":
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
                if self.prompt_resize is None:
                    region = self.resize(region)
                else:
                    region = self.prompt_resize(region)
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                    if self.prompt_resize is None:
                        region = processor.preprocess(region, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                    else:
                        region = self.prompt_processor.preprocess(region, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    if self.prompt_resize is None:
                        region = processor.preprocess(region, return_tensors='pt')['pixel_values'][0]
                    else:
                        region = self.prompt_processor.preprocess(region, return_tensors='pt')['pixel_values'][0]
            else:
                if self.debug:
                    image = processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            has_image = True
            has_region = True
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            has_image = False
            has_region = False
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
        
        if has_region:
            data_dict["region"] = region
        
        #Debug Mode. Check whether the normalization is right.
        if self.debug or (i in range(10)):
            toPIL = transforms.ToPILImage()
            region = toPIL(region)
            region.save("debug/{}-region.jpg".format(image_file.split("/")[-1]))
            bboxes = self.boxformat.extract(sources[0][-1]["value"])
            current_timestamp = time.time()
            local_time = time.localtime(current_timestamp)
            current_time = time.strftime("%Y-%m-%d_%H_%M_%S", local_time)
            try:
                visualization(os.path.join(image_folder, image_file), bboxes, "debug/{}-{}.jpg".format(current_time, image_file.split("/")[-1]), self.data_args.box_pattern)
            except:
                pass
        return data_dict