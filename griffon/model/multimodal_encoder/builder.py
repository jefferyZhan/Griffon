import os
from .clip_encoder import CLIPVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    # if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
    #     return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    # elif is_absolute_path_exists or vision_tower.startswith("sam"):
    #     return
    if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif is_absolute_path_exists:
        if "clip" in vision_tower.lower():
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    raise ValueError(f'Unknown vision tower: {vision_tower}')
