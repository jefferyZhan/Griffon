"""
ODinW dataset loading and processing utilities.
"""
import os
import math
from typing import Dict, List, Tuple
from pycocotools.coco import COCO


def round_by_factor(number: int, factor: int) -> int:
    """Return the nearest integer divisible by factor."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Return the ceiling integer divisible by factor."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Return the floor integer divisible by factor."""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int = 28, 
                 min_pixels: int = 56*56, max_pixels: int = 14*14*4*1280, 
                 max_long_side: int = 8192) -> Tuple[int, int]:
    """Resize image to meet the following conditions:
        1. Both height and width are divisible by factor
        2. Total pixels are within [min_pixels, max_pixels]
        3. Longest side is within max_long_side
        4. Aspect ratio is preserved
    
    Args:
        height: Original image height
        width: Original image width
        factor: Size must be divisible by this factor
        min_pixels: Minimum pixel count
        max_pixels: Maximum pixel count
        max_long_side: Maximum longest side
    
    Returns:
        (resized_height, resized_width): Resized dimensions
    """
    if height < 2 or width < 2:
        raise ValueError(f'height:{height} or width:{width} must be larger than factor:{factor}')
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f'absolute aspect ratio must be smaller than 200, got {height} / {width}')

    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    
    return h_bar, w_bar


def load_odinw_config(config_path: str) -> Dict:
    """Load odinw13_config.py configuration file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        datasets: Dictionary mapping dataset names to configurations
    """
    import runpy
    config = runpy.run_path(config_path)
    dataset_configs = config["datasets"]
    dataset_names = config["dataset_prefixes"]
    
    datasets = {}
    for dataset_name, dataset_config in zip(dataset_names, dataset_configs):
        datasets[dataset_name] = dataset_config
    
    return datasets


def generate_odinw_jobs(data_dir: str, args) -> Tuple[List[Dict], Dict]:
    """Generate inference task list for ODinW dataset.
    
    Args:
        data_dir: Data directory path (containing odinw13_config.py)
        args: Command line arguments
    
    Returns:
        (question_list, datasets): Task list and dataset configurations
    """
    # Load config
    config_path = os.path.join(data_dir, "dataset_config.py")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    datasets = load_odinw_config(config_path)
    
    question_list = []
    question_id = 0
    num_questions_per_dataset = {}
    
    # Calculate image resolution parameters
    patch_size = 16
    merge_base = 2
    pixels_per_token = patch_size * patch_size * merge_base * merge_base
    min_pixels = pixels_per_token * 768
    max_pixels = pixels_per_token * 12800
    
    # Iterate through all datasets
    for data_name, data_config in datasets.items():
        print(f'Parsing ODinW:{data_name}')
        classes = list(data_config["metainfo"]["classes"])
        import pdb; pdb.set_trace()
        # Build data paths
        if "odinw" in data_config["data_root"]:
            idx = data_config["data_root"].find('odinw/') + len('odinw/')
        elif "coco" in data_config["data_root"]:
            idx = data_config["data_root"].find('coco/') + len('coco/')
        elif "crowdhuman" in data_config["data_root"]:
            idx = data_config["data_root"].find('crowdhuman/') + len('crowdhuman/')
        else:
            raise AssertionError("Unsupported Task")
        sub_root = os.path.join(data_dir, data_config["data_root"][idx:])
        sub_anno = os.path.join(sub_root, data_config["ann_file"])
        sub_img_root = os.path.join(sub_root, data_config["data_prefix"]["img"])
        
        # Load COCO format annotations
        dataset = COCO(sub_anno)
        num_questions = 0
        
        # Iterate through all images
        for img_idx, img_meta in dataset.imgs.items():
            img_name = img_meta["file_name"]
            img_path = os.path.join(sub_img_root, img_name)
            img_h = img_meta["height"]
            img_w = img_meta["width"]
            
            # Calculate resized image dimensions
            resized_h, resized_w = smart_resize(
                img_h, img_w, 
                factor=32, 
                min_pixels=min_pixels, 
                max_pixels=max_pixels, 
                max_long_side=50000
            )
            
            # Get annotations
            img_annos = dataset.imgToAnns[img_idx]
            
            # Build class names list
            obj_names = ", ".join(classes)
            
            # Build prompt
            prompt = f"Locate every instance that belongs to the following categories: '{obj_names}'. Report bbox coordinates in JSON format."
            
            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image", 
                            "image": f"file://{img_path}", 
                            "min_pixels": min_pixels, 
                            "max_pixels": max_pixels
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Build task item
            item = {
                "question_id": question_id,
                "annotation": img_annos,
                'messages': messages,
                "extra_info": {
                    'dataset_name': data_name,
                    'dataset_config': data_config,
                    'img_id': img_meta["id"],
                    'anno_path': sub_anno,
                    'resized_h': resized_h,
                    'resized_w': resized_w,
                    'img_h': img_h,
                    'img_w': img_w,
                    'img_path': img_path
                }
            }
            question_list.append(item)
            question_id += 1
            num_questions += 1
        
        num_questions_per_dataset[data_name] = num_questions
    
    # Print statistics
    for data_name, num_questions in num_questions_per_dataset.items():
        print(f'{data_name}: {num_questions}')
    print(f"Total ODinW questions: {len(question_list)}")
    
    return question_list, datasets