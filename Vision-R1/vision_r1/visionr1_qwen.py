# Copyright 2025 Griffon Team.

import os
import re
import ast
import json
import torch
import copy
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from functools import partial
from typing import Optional
from scipy.optimize import linear_sum_assignment
from numpy import mean

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration
from transformers import TrainingArguments

from math_verify import parse, verify
from vision_r1.trainer import Qwen25VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from griffon.constants import DEFAULT_IMAGE_TOKEN
from griffon.coor_utils import generalized_box_iou, accum_probs, box_iou



def parse_json(json_output):
    """
    parse output
    """
    try:
        lines = json_output.strip().splitlines()
        
        # remove Markdown JSON tag
        if lines[0].startswith("```json"):
            json_output = "\n".join(lines[1:])  # remove "```json"
        
        if "```" in json_output:
            json_output = json_output.split("```")[0]  # remove ```
        
        return json.loads(json_output)  # parse JSON
    except Exception:
        return None  # if fail, return None

def extract_bbox_label(text):
    """
    extract  {"bbox_2d": [...], "label": "..."} by re
    """
    try:
        pattern = re.compile(r'\{"bbox_2d": \[(\d{1,3}|1000), (\d{1,3}|1000), (\d{1,3}|1000), (\d{1,3}|1000)\], "label": "([^"]+)"\}')
        matches = pattern.findall(text)

        extracted_data = []
        for match in matches:
            x1, y1, x2, y2, label = match
            bbox = [int(x1), int(y1), int(x2), int(y2)]

            # filter negative bbox（check x1 < x2, y1 < y2）
            if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                extracted_data.append({"bbox": bbox, "category_name": label})

        return extracted_data
    except Exception:
        return []  # return blank

def extract(completion):
    """
    Extract bbox_2d and lable, and return blank list if failed
    """
    output = []
    try:
        extracted = parse_json(completion)
        if isinstance(extracted, list):  # ensure list output
            for item in extracted:
                try:
                    category_name = item.get("label")
                    bbox = item.get("bbox_2d")

                    # ensure bbox format
                    if isinstance(bbox, list) and len(bbox) == 4:
                        bbox = [int(coord) for coord in bbox]  # ensure int
                        if bbox[0] < bbox[2] and bbox[1] < bbox[3]:  # filter negative bbox
                            output.append({"bbox": bbox, "category_name": category_name})
                except Exception:
                    continue  # ignore single error and continue
    except Exception:
        pass  # if failed, turn to re match

    if not output:  # if failed, turn to re match
        try:
            output = extract_bbox_label(completion)
        except Exception:
            output = []  

    return output  
    

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["precision_reward", "dual_format_reward", "recall_reward"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    hard: Optional[bool] = field(
        default=False,
        metadata={"help": "whether filter the instance to over 10"}
    )
    eval_match: Optional[bool] = field(default=False)
    adaptive: Optional[bool] = field(default=False)


def BoxPriorMatcher(outputs, targets):
    """ 
    Performs the matching, ignoring the possible but seldom occured class prediction error and focusing on the box quality instead.
    If the predicted ones are less than the GTs, each prediction will have a matched GT box. Otherwise, only num_of_target_bbox 
    predictions will have matched GTs, i.e. IoUs.

    Params:
        outputs: This is a dict that contains at least these entries:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

        targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                        objects in the target) containing the class labels
                "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

    Intermediate Output:
        A list of size batch_size, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
        For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)

    Return:
        A reward list of size [batch_size, min(num_queries, num_gt)]
    """
    bs, num_queries = outputs["pred_logits"].shape[:2]

    # We flatten to compute the cost matrices in a batch
    out_prob = outputs["pred_logits"].flatten(0, 1)  # [batch_size * num_queries, num_classes]
    out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

    # Also concat the target labels and boxes
    tgt_ids = torch.cat([v["labels"] for v in targets])
    tgt_bbox = torch.cat([v["boxes"] for v in targets])

    cost_class = -out_prob[:, tgt_ids]

    # Compute the L1 cost between boxes for matching
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # w/o normalize the bbox to prior bbox 

    # Compute the giou cost betwen boxes for reward
    cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

    # Final cost matrix 
    C = cost_bbox + cost_class + cost_giou
    C = C.view(bs, num_queries, -1).cpu()

    sizes = [len(v["boxes"]) for v in targets]
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

    if cost_giou.ndim == 3:
        returned_iou = []
        for cost_giou_, index in zip(cost_giou, indices):
            returned_iou.append(-cost_giou_[index])
    else:
        returned_iou = -cost_giou[indices[0]]
        
    return returned_iou

def modify_list(lst, minimal=0.5, maximum=0.75):
    """
    Change to adapative, when model can well perform than one threshold, change it to a higher value
    """
    return [0 if x < minimal else 1 if x > maximum else x for x in lst]
    

def dual_format_reward(completions, num_instances, **kwargs):
    def validate_bbox_label(text):
        # remove Markdown  ```json 和 ```
        text = re.sub(r'```json\n|\n```', '', text).strip()
        reward = 1.0
        # try to parse JSON
        try:
            data = json.loads(text)  
            if isinstance(data, list):  
                for item in data:
                    if isinstance(item, dict) and "bbox_2d" in item and "label" in item:
                        bbox = item["bbox_2d"]
                        
                        if not (
                        isinstance(bbox, list) and len(bbox) == 4):
                            reward = 0.0  
        except json.JSONDecodeError:
            reward = 0.0 

        return reward

    rewards = []
    
    for item, num_instance in zip(completions, num_instances):
        item = item[0]["content"]
        reward = validate_bbox_label(item)
        
        if num_instance > 0:
            rewards.append(reward)
        else:
            rewards.append(1.0 - reward)

    return rewards

def modify_tensor(tensor, minimal=0.5, maximum=0.75):
    tensor = tensor.float()
    
    tensor[tensor < minimal] = 0
    
    tensor[tensor > maximum] = 1
    
    return tensor

def recall_reward(completions, solution, width, height, num_instances, step, input_widths, input_heights, **kwargs):
    """
    Reward function that check whether the completion contain the exact number of instances
    Previous version mainly focus on 
    """
    if step is not None:
        modify_list_ = partial(modify_list, minimal=0.5, maximum=0.75)
    else:
        modify_list_ = partial(modify_list, minimal=0, maximum=1)

    if step is None or step < 1500:
        modify_match_ = partial(modify_tensor, minimal=0.5, maximum=0.5)
    else:
        modify_match_ = partial(modify_tensor, minimal=0.75, maximum=0.75) 

    solutions = [json.loads(item) for item in solution]
    
    target_id_maps = kwargs.pop("target_id_maps", None)
    if target_id_maps:
        target_id_maps = [json.loads(item) for item in target_id_maps]
    else:
        raise AssertionError
    middle_brackets_pat = re.compile("(\[\d{1,}\.\d{3},\d{1,}\.\d{3},\d{1,}\.\d{3},\d{1,}\.\d{3}\])")
    rewards = []
    for completion, solution, width, height, target_id_map, num_instance, input_height, input_width in zip(completions, solutions, width, height, target_id_maps, num_instances, input_heights, input_widths):
        # process dt
        completion = completion[0]["content"].strip()
        dt = extract(completion)
        target_id_map = {key.lower(): value for key, value in target_id_map.items()}

        if len(dt):
            dt_class_id = []
            dt_bboxes = []
            for item in dt:
                try:
                    class_id = target_id_map[item["category_name"].strip().lower()]
                except:
                    class_id = 0
                pred_bbox = torch.tensor(item["bbox"], dtype=torch.float32).view(-1, 4)
                pred_bbox[:, ::2] = pred_bbox[:, ::2] / input_width * width
                pred_bbox[:, 1::2] = pred_bbox[:, 1::2] / input_height * height
                dt_class_id.append(class_id)
                dt_bboxes.append(pred_bbox)
            dt_bboxes = torch.cat(dt_bboxes)
            dt_class = torch.zeros([dt_bboxes.size()[0],max(target_id_map.values())+1], dtype=torch.float32)# +1 due to set the class 0 to the background
            dt_class[torch.arange(dt_class.shape[0]), dt_class_id] = 1.0
            dt_dict = {
                "pred_logits": dt_class.unsqueeze(0),
                "pred_boxes": dt_bboxes.unsqueeze(0)
            }
            #process gt
            gt_class_id = []
            gt_bboxes = []
            for item in solution:
                class_id, bbox = item
                pred_bbox = torch.tensor(bbox, dtype=torch.float32).view(-1, 4)
                gt_class_id.append(int(target_id_map[class_id.lower()]))
                gt_bboxes.append(pred_bbox)
            gt_bboxes = torch.cat(gt_bboxes)
            gt_dict = {
                "labels": torch.tensor(gt_class_id),
                "boxes": gt_bboxes
            }
            reward_of_each_instance = BoxPriorMatcher(dt_dict, [gt_dict])
            tp = sum(modify_match_(reward_of_each_instance))
            recall = tp/num_instance
            recall = modify_list_([recall])[0] # Reward and Penity for Recall
            rewards.append(recall)
        else:
            if num_instance > 0: # 
                rewards.append(0.0)
            else:
                rewards.append(1.0)
    return rewards

def precision_reward(completions, solution, width, height, input_heights, input_widths, scores=None, ids=None, step=None, **kwargs):
    """
    Reward function used to calculate the localization accuracy of each instance
    First, we consider all the instances the same quality due to the limited predictions
    """
    if step is not None:
        if step < 1500:
            modify_list_ = partial(modify_list, minimal=0.5, maximum=0.75)
        else:
            modify_list_ = partial(modify_list, minimal=0.75, maximum=0.9)
    else:
        modify_list_ = partial(modify_list, minimal=0, maximum=1)

    if step is not None:
        if step is None or step < 1500:
            modify_match_ = partial(modify_tensor, minimal=0.5, maximum=0.75)
        else:
            modify_match_ = partial(modify_tensor, minimal=0.75, maximum=0.9) 
    else:
        modify_match_ = partial(modify_tensor, minimal=0, maximum=1)

    solutions = [json.loads(item) for item in solution]
    # FIX
    target_id_maps = kwargs.pop("target_id_maps", None)
    if target_id_maps:
        target_id_maps = [json.loads(item) for item in target_id_maps]
    else:
        raise AssertionError
    num_instances = kwargs.pop("num_instances", None)

    #target_id_maps = len(solutions) * [TARGET] # each prompt categories to ids map
    middle_brackets_pat = re.compile("(\[\d{1,}\.\d{3},\d{1,}\.\d{3},\d{1,}\.\d{3},\d{1,}\.\d{3}\])")
    rewards = []


    for completion, solution, width, height, target_id_map, num_instance, input_height, input_width in zip(completions, solutions, width, height, target_id_maps, num_instances, input_heights, input_widths):
        # process dt
        completion = completion[0]["content"].strip()
        dt = extract(completion)
        target_id_map = {key.lower(): value for key, value in target_id_map.items()}

        if len(dt):
            dt_class_id = []
            dt_bboxes = []
            for item in dt:
                try:
                    class_id = target_id_map[item["category_name"].strip().lower()]
                except:
                    class_id = 0
                pred_bbox = torch.tensor(item["bbox"], dtype=torch.float32).view(-1, 4)
                pred_bbox[:, ::2] = pred_bbox[:, ::2] / input_width * width
                pred_bbox[:, 1::2] = pred_bbox[:, 1::2] / input_height * height
                dt_class_id.append(class_id)
                dt_bboxes.append(pred_bbox)
            dt_bboxes = torch.cat(dt_bboxes)
            dt_class = torch.zeros([dt_bboxes.size()[0],max(target_id_map.values())+1], dtype=torch.float32)# +1 due to set the class 0 to the background
            try:
                dt_class[torch.arange(dt_class.shape[0]), dt_class_id] = 1.0
            except:
                print(dt_class_id, target_id_map)
                dt_class[torch.arange(dt_class.shape[0]), dt_class_id] = 1.0
            dt_dict = {
                "pred_logits": dt_class.unsqueeze(0),
                "pred_boxes": dt_bboxes.unsqueeze(0)
            }
            #process gt
            gt_class_id = []
            gt_bboxes = []
            for item in solution:
                class_id, bbox = item
                pred_bbox = torch.tensor(bbox, dtype=torch.float32).view(-1, 4)
                gt_class_id.append(int(target_id_map[class_id.lower()]))
                gt_bboxes.append(pred_bbox)
            gt_bboxes = torch.cat(gt_bboxes)
            gt_dict = {
                "labels": torch.tensor(gt_class_id),
                "boxes": gt_bboxes
            }

            reward_of_each_instance = BoxPriorMatcher(dt_dict, [gt_dict])
            reward_of_each_instance = modify_match_(reward_of_each_instance)
            rewards.append(torch.mean(reward_of_each_instance).item())
        else:
            if num_instance > 0: 
                rewards.append(0.0)
            else:
                rewards.append(1.0)
    
    return rewards

reward_funcs_registry = {
    "dual_format_reward": dual_format_reward,
    "precision_reward": precision_reward,
    "recall_reward": recall_reward,
}


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    try:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    except:
        dataset = load_from_disk(script_args.dataset_name)

    def make_conversation_image(example, qwen_enable):
        # change the prompt to qwen mode
        if qwen_enable:
            if "Examine the image for any objects from the category set. Report the coordinates of each detected object." in example["problem"]:
                pb = example["problem"].replace("Examine the image for any objects from the category set. Report the coordinates of each detected object.", "Locate every item from the category list in the image and output the coordinates in JSON format.")
                #pb = "Please output bbox coordinates and names of every item in this image."
            elif "Can you point out" in example["problem"]:
                pb = example["problem"].replace("Can you point out", "Locate every").replace("in the image and provide the coordinates of its location?", "in the image and output the coordinates in JSON format.")
            elif "Locate the exact position of" in example["problem"]:
                pb = example["problem"].replace("Locate the exact position of", "Locate every").replace("in the picture, if you can.", "in the image and output the coordinates in JSON format.")
            else:
                raise AssertionError
        else:
            pb = example["problem"]
        
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": pb},
                    ],
                },
            ],
        }
    
    def filter_has_instance(example, minimal=1, maximum=30):
        return int(example["num_instances"])>=minimal and int(example["num_instances"])<=maximum
    
    if "qwen" in model_args.model_name_or_path.lower():
        print("Qwen Enabled")
        make_conversation_image = partial(make_conversation_image, qwen_enable=True)

    if not script_args.hard:
        dataset = dataset.filter(filter_has_instance)
    else:
        dataset = dataset.filter(partial(filter_has_instance,minimal=10))
    dataset = dataset.sort("num_instances")

    if "image" in dataset.features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping

    else:
        print("no image in dataset")
        raise AssertionError
    
    trainer_cls = Qwen25VLGRPOTrainer


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        adapative=script_args.adaptive,
        torch_dtype=model_args.torch_dtype
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
