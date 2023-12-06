import os
from .clip_encoder import CLIPVisionTower
from llava.datasets.base_class import rank0_print


def build_vision_tower(vision_tower_cfg, **kwargs):
    #vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    vision_tower = getattr(vision_tower_cfg, 'vision_tower', getattr(vision_tower_cfg, 'mm_vision_tower', None))
    #vision_tower = "/tmp/clip-vit-large-patch14-448"
    is_absolute_path_exists = os.path.exists(vision_tower)
    # rank0_print(f"Vision tower is {vision_tower}. Exist: {is_absolute_path_exists}")
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    raise ValueError(f'Unknown vision tower: {vision_tower}')
