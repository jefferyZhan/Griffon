import re, json
import torch
from functools import partial
from typing import Optional, Any
from torchvision.ops.boxes import box_area
from scipy.optimize import linear_sum_assignment

REWARD_NAME = "visionr1"
REWARD_TYPE = "sequential_ground" #Modify: support different reward type

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

def modify_tensor(tensor, minimal=0.5, maximum=0.75):
    tensor = tensor.float()
    
    tensor[tensor < minimal] = 0
    
    tensor[tensor > maximum] = 1
    
    return tensor
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

def dual_format_reward(completion, num_instance):
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
    
    reward = validate_bbox_label(completion)
    
    if num_instance > 0:
        return reward
    else:
        return (1.0 - reward)

def recall_reward(completion, solution, width, height, target_id_map, num_instance, input_width=None, input_height=None, step=None):
    """
    Reward function that check whether the completion contain the exact number of instances
    Previous version mainly focus on 
    """
    if input_width is None or input_height is None:
        # relative
        input_width = 1000
        input_height = 1000
    else:
        # absolute
        pass
    if step is not None:
        modify_list_ = partial(modify_list, minimal=0.5, maximum=0.75)
    else:
        modify_list_ = partial(modify_list, minimal=0, maximum=1)

    if step is None or step < 1500:
        modify_match_ = partial(modify_tensor, minimal=0.5, maximum=0.5)
    else:
        modify_match_ = partial(modify_tensor, minimal=0.75, maximum=0.75) 

    try:
        solution = json.loads(solution)
        target_id_map = json.loads(target_id_map)
    except:
        raise AssertionError(f"Either solution or target_id_map is error, please check!\n solution: {solution}\n target_id_map: {target_id_map}")
    
    # process dt
    dt = extract(completion.strip())
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
        reward = recall
    else:
        if num_instance > 0: # 
            reward = 0.0
        else:
            reward = 1.0
            
    return reward

def precision_reward(completion, solution, width, height, target_id_map, num_instance, input_width=None, input_height=None, step=None):
    """
    Reward function used to calculate the localization accuracy of each instance
    First, we consider all the instances the same quality due to the limited predictions
    """
    if input_width is None or input_height is None:
        # relative
        input_width = 1000
        input_height = 1000
    else:
        # absolute
        pass

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

    # FIX
    try:
        solution = json.loads(solution)
        target_id_map = json.loads(target_id_map)
    except:
        raise AssertionError(f"Either solution or target_id_map is error, please check!\n solution: {solution}\n target_id_map: {target_id_map}")

    #target_id_maps = len(solutions) * [TARGET] # each prompt categories to ids map
    middle_brackets_pat = re.compile("(\[\d{1,}\.\d{3},\d{1,}\.\d{3},\d{1,}\.\d{3},\d{1,}\.\d{3}\])")

    # process dt
    dt = extract(completion.strip())
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
        # Update
        reward = torch.sum(reward_of_each_instance).item() / len(dt)
    else:
        if num_instance > 0: 
            reward = 0.0
        else:
            reward = 1.0
    
    return reward

def compute_score(reward_input: dict[str: Any], format_weight: float = 1/3) -> dict[str, float]:
    #completion, solution, width, height, target_id_map, num_instance, input_width=None, input_height=None, step=None
    predict_str = reward_input["response"]
    ground_truth = reward_input["solution"]
    width = reward_input["width"]
    height = reward_input["height"]
    target_id_maps = reward_input["target_id_maps"]
    num_instance = reward_input["num_instances"]

    try:
        recall_score = recall_reward(predict_str, ground_truth, width, height, target_id_maps, num_instance)
        precision_score = precision_reward(predict_str, ground_truth, width, height, target_id_maps, num_instance)
        format_score = dual_format_reward(predict_str, num_instance)
        if torch.is_tensor(recall_score):
            recall_score = recall_score.item()
        if torch.is_tensor(precision_score):
            precision_score = precision_score.item()
        score = ((precision_score + recall_score + format_score) / 3.0)
    except:
        torch.save(reward_input, "/mnt/bn/ecomckptnas/yufeizhan/valleyomni_training_yamls/TRY/EasyR1/test.pth")
        raise AssertionError(f"Computation Error. Data saved!")

    return {"overall": score, "format": format_score, "recall": recall_score, "precision": precision_score}

def compute_score_precision_only(reward_input: dict[str: Any], format_weight: float = 1/3) -> dict[str, float]:
    #completion, solution, width, height, target_id_map, num_instance, input_width=None, input_height=None, step=None
    predict_str = reward_input["response"]
    ground_truth = reward_input["solution"]
    width = reward_input["width"]
    height = reward_input["height"]
    target_id_maps = reward_input["target_id_maps"]
    num_instance = reward_input["num_instances"]

    try:
        #recall_score = recall_reward(predict_str, ground_truth, width, height, target_id_maps, num_instance)
        precision_score = precision_reward(predict_str, ground_truth, width, height, target_id_maps, num_instance)
        format_score = dual_format_reward(predict_str, num_instance)
        # if torch.is_tensor(recall_score):
        #     recall_score = recall_score.item()
        if torch.is_tensor(precision_score):
            precision_score = precision_score.item()
        score = ((precision_score + format_score) / 2.0)
    except:
        torch.save(reward_input, "./test.pth")
        raise AssertionError(f"Computation Error. Data saved!")

    return {"overall": score, "format": format_score, "precision": precision_score}

if __name__ == "__main__":
    reward_input = torch.load("./test.pth")
    import pdb; pdb.set_trace()
    score = compute_score(reward_input)
    print(score)