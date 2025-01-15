import os, json
from typing import Dict, List, Optional, Union
import torch
import argparse
import transformers
import itertools
import copy
import re

from torch.utils.data import Dataset
from tqdm import tqdm
from functools import partial
import numpy as np
from torchvision.ops.boxes import box_area
from PIL import Image
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from torchvision import transforms

from griffon.eval.run_griffon import load_image, extract
from griffon.coor_utils import xyxy2xywh, accum_probs
from griffon.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from griffon.conversation import conv_templates, SeparatorStyle
from griffon.model.builder import load_pretrained_model
from griffon.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from griffon.utils import auto_rank0_print

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

class EVALDataset(Dataset):
    def __init__(self, data_path: str, prompt: str, image_folder: str, tokenizer: transformers.PreTrainedTokenizer,
                 image_processor):
        super(EVALDataset, self).__init__()
        f = open(data_path, "r", encoding="utf-8")
        list_data_dict = f.readlines()
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.image_processor = image_processor
        self.prompt = prompt
        self.image_folder = image_folder
        self.resize = transforms.Resize((image_processor.size["shortest_edge"], image_processor.size["shortest_edge"]))

    def __len__(self):
        return len(self.list_data_dict)

    def get_item(self, index):
        """
            return {
                "image_path":
                "gt_bbox": transformered
            }
        """
        source = self.list_data_dict[index]
        item = json.loads(source)
        img_path = item["img_path"]
        expr = item["expression"]
        bbox = item["bbox"]
        width = item["width"]
        height = item["height"]
        query = self.prompt.replace("<expr>", expr)
        ret = {
            "image_path": img_path,
            "query": query,
            "gt_bbox": bbox,
            "height": height,
            "width": width
        }
        return ret
    
    def __getitem__(self, index):
        source = self.get_item(index)
        # process image
        image_path = os.path.join(self.image_folder, source["image_path"])
        img = load_image(image_path)
        img = self.resize(img)
        img_tensor = self.image_processor.preprocess(img, return_tensors='pt')['pixel_values'] #TODO：修改为仅进行resize
        ret = {
            "image_tensor": img_tensor,
            "gt_bbox": source["gt_bbox"],
            "query": source["query"],
            "height": source["height"],
            "width": source["width"],
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

def collate_fn(batches, tokenizer, conv_mode):
    # dataloader会将数据自动转为cuda
    input_ids = []
    bboxes = []
    image_tensors = []
    infos = []
    for _ in batches:
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], _["query"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids.append(tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'))
        bboxes.append(_["gt_bbox"])
        image_tensors.append(_["image_tensor"])
        info = {"height": _["height"], "width": _["width"]}
        infos.append(info)

    # TODO: Support batch size>1
    return input_ids[0].unsqueeze(0), image_tensors[0], bboxes, infos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="JefferyZhan/Griffon-G-gemma2-9b")
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--query", type=str, default="Can you point out <expr> in the image and provide the coordinates of its location?")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument('--dataset', type=str, help="Path to refcoco annotations", required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--output-path', type=str, default="./eval_output")
    parser.add_argument('--init',type=str, default="tcp://127.0.0.1:12457")
    args = parser.parse_args()

    # Env Init
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
        init_method=str(args.init)
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    #Load Model & Model init
    model_name = get_model_name_from_path(args.model_path)
    if "llama" in model_name.lower():
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name, torch_dtype=torch_dtype, device_map="cuda")

    prompt = args.query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    if 'llama' in model_name.lower():
        conv_mode = "llava_llama_2"
    else:
        conv_mode = "gemma_instruct"
    
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode
    
    model_out_dir = os.path.join(args.output_path, args.model_path.split("/")[-1])
    os.makedirs(model_out_dir, exist_ok=True)

    datasets = [
        "REC_refcoco_unc_testA.jsonl",
        "REC_refcocog_umd_test.jsonl",
        "REC_refcoco+_unc_testA.jsonl",  
        "REC_refcoco_unc_val.jsonl",
        "REC_refcocog_umd_val.jsonl", 
        "REC_refcoco_unc_testB.jsonl", 
        "REC_refcoco+_unc_val.jsonl", 
        "REC_refcoco+_unc_testB.jsonl",
    ]
    model_data_out_dir = os.path.join(model_out_dir, "RefCOCO")
    os.makedirs(model_data_out_dir, exist_ok=True)
    print(args)
    for dataset_ in datasets:
        dataset_path = os.path.join(args.dataset, dataset_)
        dataset_name = dataset_.split(".")[0]
        dataset = EVALDataset(dataset_path, qs, args.image_folder, tokenizer, image_processor)
    
        conv = conv_templates[args.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler = InferenceSampler(len(dataset)),
            batch_size = args.batch_size,
            num_workers = args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer, conv_mode=args.conv_mode)
        )

        if not os.path.exists(os.path.join(model_data_out_dir, dataset_.replace(".jsonl", "_output.jsonl"))):
            eval_outputs = []
            with torch.inference_mode():
                for i, (input_ids, image_tensors, bboxes, infos) in tqdm(enumerate(dataloader)):
                    input_ids = input_ids.cuda()
                    image_tensors = image_tensors.half().cuda()
                    assert args.batch_size == 1
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                    
                    output_dict = model.generate(
                            input_ids,
                            images=image_tensors.to(dtype=torch_dtype, device='cuda', non_blocking=True),
                            num_beams=1,
                            max_new_tokens=128,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                            return_dict_in_generate=True,
                            output_scores=True
                        )
                    output_ids = output_dict.sequences
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
                    
                    for bbox, info, output in zip(bboxes, infos, [outputs]):
                        eval_outputs.append({
                            "output": output,
                            "gt_box": bbox,
                            "height": info["height"],
                            "width": info["width"]
                        })

                torch.distributed.barrier()
                world_size = torch.distributed.get_world_size()
                merged_outputs = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(merged_outputs, eval_outputs)
                merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

            if torch.distributed.get_rank() == 0:
                save_output = [json.dumps(item)+"\n" for item in merged_outputs]
                file = open(os.path.join(model_data_out_dir, dataset_.replace(".jsonl", "_output.jsonl")), "w")
                file.writelines(save_output)
        else:
            if torch.distributed.get_rank() == 0:
                merged_outputs = [json.loads(line) for line in open(os.path.join(model_data_out_dir, dataset_.replace(".jsonl", "_output.jsonl"))).readlines()]
        if torch.distributed.get_rank() == 0:
            print("Begin to eval")
            correct = 0
            total = 0
            middle_brackets_pat = re.compile("(\[\d{1,}\.\d{3},\d{1,}\.\d{3},\d{1,}\.\d{3},\d{1,}\.\d{3}\])")
            for _, output in tqdm(enumerate(merged_outputs)):
                total += 1
                try:
                    bbox = output["output"]
                    if isinstance(bbox, list):
                        bbox = extract(bbox[0], middle_brackets_pat)[0]["bbox"]
                    elif isinstance(bbox, str):
                        bbox = extract(bbox, middle_brackets_pat)[0]["bbox"]
                    predict_bbox = torch.tensor(bbox, dtype=torch.float32).view(-1, 4)
                    gt_bbox = torch.tensor(output["gt_box"], dtype=torch.float32).view(-1, 4)
                    if (gt_bbox[0, 0] == 0.0 and gt_bbox[0, 2] == 0.0) or (gt_bbox[0, 0] == 1.0 and gt_bbox[0, 2] == 1.0):
                        continue
                    if (gt_bbox[0, 1] == 0.0 and gt_bbox[0, 3] == 0.0) or (gt_bbox[0, 1] == 1.0 and gt_bbox[0, 3] == 1.0):
                        continue
                    predict_bbox[:, 0::2] *= output["width"]
                    predict_bbox[:, 1::2] *= output["height"]
                    iou, _ = box_iou(predict_bbox, gt_bbox)
                    if iou > 0.5:
                        correct += 1
                except:
                    continue
            acc = correct / total
            print("{}: {:.3f}".format(dataset_name, acc))
        torch.distributed.barrier()