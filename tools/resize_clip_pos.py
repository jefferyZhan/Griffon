import torch
import collections
import math
import logging
import torch.nn.functional as F
import argparse
import copy
import shutil
import os

from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

def resize_pos_embed(state_dict, new_size, patch_size, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    # postion_embedding = [cls_pos_emb, patch_pos_emb]
    # key name: vision_model.embeddings.position_ids vision_model.embeddings.position_embedding.weight
    #import pdb; pdb.set_trace()
    assert 'vision_model.embeddings.position_embedding.weight' in state_dict and "vision_model.embeddings.position_ids" in state_dict
    old_pos_embed = state_dict.pop('vision_model.embeddings.position_embedding.weight')
    assert old_pos_embed is not None
    grid_size = new_size // patch_size
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size * grid_size + extra_tokens
    assert new_seq_len != old_pos_embed.shape[0], "No Need to resize positional embedding"

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size * grid_size, -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    
    logging.info(f"Resize the position_ids from {len(pos_emb_img)} to {new_seq_len}")
    new_position_ids = torch.arange(new_seq_len).expand((1, -1))
    state_dict["vision_model.embeddings.position_ids"] = new_position_ids.type(state_dict["vision_model.embeddings.position_ids"].dtype).cpu()
    state_dict['vision_model.embeddings.position_embedding.weight'] = new_pos_embed.type(old_pos_embed.dtype).cpu()
    return state_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--new-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    # parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()
    if args.model_path.endswith("/"):
        target_path = args.model_path[:-1] + f"_to_{args.new_size}"
    else:
        target_path = args.model_path + f"_to_{args.new_size}"
    # os.mkdir(target_path)
    shutil.copytree(args.model_path, target_path)
    target_model_path = os.path.join(target_path, "pytorch_model.bin")
    os.remove(target_model_path)
    state_dict = copy.deepcopy(torch.load(os.path.join(args.model_path, "pytorch_model.bin"), map_location="cpu"))
    state_dict = resize_pos_embed(state_dict, args.new_size, args.patch_size)
    torch.save(state_dict, target_model_path)
    print("Finish updating the position embedding. Please modify the .json file according to your customization and the guideline.")
