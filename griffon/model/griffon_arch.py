#    Copyright 2024 Yufei Zhan
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import math
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from griffon.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from transformers import CLIPVisionModel
from .utils import LayerNorm

from griffon.utils import auto_rank0_print

class GriffonMetaModel:
    def __init__(self, config):
        super(GriffonMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)
        if getattr(config, "unshared_visual_encoder", False):
            self.prompt_visual_encoder = CLIPVisionModel.from_pretrained("checkpoints/clip-vit-large-patch14", device_map="auto")
            self.prompt_projector = nn.Sequential(
                nn.Linear(self.config.mm_hidden_size, self.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size, self.config.hidden_size)
            )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            auto_rank0_print("vision encoder is not loaded, and is loaded by initialization.")
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            if not vision_tower.is_loaded:
                auto_rank0_print("load vision model from original checkpoint.")
                vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.unshared_visual_encoder = model_args.unshared_visual_encoder
        
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
        
        #Modify 
        if model_args.unshared_visual_encoder:
            if getattr(self, 'prompt_visual_encoder', None) is None:
                self.prompt_visual_encoder = CLIPVisionModel.from_pretrained("checkpoints/clip-vit-large-patch14") #224 encoder
                for p in self.prompt_visual_encoder.parameters():
                    p.requires_grad = False
                self.prompt_projector = nn.Sequential(
                    nn.Linear(self.config.mm_hidden_size, self.config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.config.hidden_size, self.config.hidden_size)
                )
                for p in self.prompt_projector.parameters():
                    #train the projector
                    p.requires_grad = True
            else:
                for p in self.prompt_visual_encoder.parameters():
                    p.requires_grad = False
                for p in self.prompt_projector.parameters():
                    #train the projector
                    p.requires_grad = True

class GriffonMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def set_image_tokens(self, image_tokens):
        # import pdb; pdb.set_trace()
        bs, grid2, C = image_tokens.shape
        image_tokens = image_tokens.detach().permute(0,2,1).view(bs, C, int(grid2**0.5), int(grid2**0.5))
        #bottom_up_feature = self.get_model().image_token_reg_bottomup(image_tokens).flatten(2,3).permute(0,2,1)
        bottom_up_feature = self.get_model().image_reg_bottomup(image_tokens).flatten(2,3).permute(0,2,1)
        self.image_tokens_input = self.get_model().image_reg_input(bottom_up_feature)
        self.image_tokens_output = self.get_model().image_reg_output(bottom_up_feature)

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # import pdb; pdb.set_trace()
        if getattr(self.config, "image_token_reg", False):
            self.set_image_tokens(image_features)
        image_features = self.get_model().mm_projector(image_features)
        
        return image_features
    
    def encode_numbers(self, new_input_embeds, new_labels, numbers, number_token_index = 32000):
        number_mask = torch.where(new_labels == number_token_index)
        numbers_tensor = torch.cat([torch.tensor(number) for i, number in enumerate(numbers)]).view(-1).to(new_input_embeds.device)
        return numbers_tensor, number_mask

    def encoder_prompts(self, images):
        def feature_select(hidden_states, selected="cls"):
            features = hidden_states.hidden_states[-1] # for cls token
            if selected == "patch":
                return features[:, 1:]
            elif selected == "cls":
                return features[:, 0].unsqueeze(1)

        # For CLIP
        with torch.no_grad():
            if type(images) is list:
                image_features = []
                for image in images:
                    image_forward_out = self.get_model().prompt_visual_encoder(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                    image_feature = feature_select(image_forward_out).to(image.dtype)
                    image_features.append(image_feature)
            else:
                image_forward_outs = self.get_model().prompt_visual_encoder(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features = feature_select(image_forward_outs).to(images.dtype)

        image_features = self.get_model().prompt_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, numbers=None, regions=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                if numbers is not None:
                    input_embeds = self.get_input_embeddings()(input_ids) 
                    input_embeds_out, numbers_tensor, number_mask = self.encode_numbers(input_embeds, input_ids, numbers)
                    return None, position_ids, attention_mask, past_key_values, input_embeds_out, labels, numbers_tensor, number_mask
                if (input_ids >= self.get_input_embeddings().num_embeddings).any():

                    assert input_ids.shape[1] == 1, "each token prediction."
                    in_embed = self.get_input_embeddings()
                    input_ids = input_ids - in_embed.num_embeddings
                    input_embeddings = self.image_tokens_input[0,input_ids.item()]
                    if input_embeddings.ndim == 1:
                        input_embeddings = input_embeddings.reshape(1,1,-1)
                    elif input_embeddings.ndim == 2:
                        input_embeddings = input_embeddings.unsqueeze(0)

                    return None, position_ids, attention_mask, past_key_values, input_embeddings, labels, None, None

            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None

        if not getattr(self.config, "unshared_visual_encoder", False):
            if type(images) is list or images.ndim == 5:
                concat_images = torch.cat([image for image in images], dim=0)
                image_features = self.encode_images(concat_images)
                split_sizes = [image.shape[0] for image in images]
                image_features = torch.split(image_features, split_sizes, dim=0)
                image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
            else:
                image_features = self.encode_images(images).to(self.device)
            prompt_image_features = None
        else:
            
            if type(images) is list or images.ndim == 5:
                
                concat_images = torch.cat([image for image in images], dim=0)
                image_features = self.encode_images(concat_images)
                split_sizes = [image.shape[0] for image in images]
                image_features = torch.split(image_features, split_sizes, dim=0)
                image_features = [x.flatten(0, 1) for x in image_features]
                if regions != None:
                    assert len(images) == len(regions), f"The number of images {len(images)} must match the number of regions{len(regions)}."
                    prompt_images = torch.cat([region for region in regions], dim=0)
                    prompt_image_features = self.encoder_prompts(prompt_images)
                    prompt_split_size = [region.shape[0] for region in regions]
                    prompt_image_features = torch.split(prompt_image_features, prompt_split_size, dim=0)
                    prompt_image_features = [x.flatten(0, 1) for x in prompt_image_features]
                else:
                    prompt_image_features = None
            else:
                image_features = self.encode_images(images)
                if regions != None:
                    assert images.shape[0] == regions.shape[0], "The number of images must match the number of regions."
                    prompt_image_features = self.encoder_prompts(regions)
                else:
                    prompt_image_features = None
            
            
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError
        

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            #import pdb; pdb.set_trace()
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_input_embeddings()(cur_input_ids) # get_model().embed_tokens -> get_input_embeddings
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            if getattr(self.config, "image_token_reg", False):
                # import pdb; pdb.set_trace()
                in_embed = self.get_input_embeddings()
                cur_input_ids = torch.cat(cur_input_ids_noim)
                mask = cur_input_ids >= in_embed.num_embeddings
                ori_ids = cur_input_ids.masked_fill(mask, 0)
                cur_input_embeds = in_embed(ori_ids)
                index = cur_input_ids[mask] - in_embed.num_embeddings
                selected_image_tokens = (self.image_tokens_input[batch_idx][index]).to(cur_input_embeds.dtype)
                cur_input_embeds.masked_scatter_(mask[:,None], selected_image_tokens)
            else:
                cur_input_embeds = self.get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    # modify
                    if prompt_image_features is not None and i % 2 == 1:
                        cur_prompt_features = prompt_image_features[cur_image_idx-1] # offset-1
                        cur_new_input_embeds.append(cur_prompt_features)
                        cur_new_labels.append(torch.full((1,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    else:
                        cur_image_features = image_features[cur_image_idx]
                        
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))


            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        ### ZYF Modify for replace the NUM embedding with number condition
        ### Modify based on the label to generate the mask
        if numbers is not None:
            numbers_, numbers_mask_ = self.encode_numbers(new_input_embeds, new_labels_padded, numbers)
        else:
            numbers_ = None
            numbers_mask_ = None

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, numbers_, numbers_mask_

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
