#    Copyright 2023 Haotian Liu
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


from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateOutput

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput
from transformers.generation.stopping_criteria import validate_stopping_criteria
from transformers.utils import ModelOutput

from ..griffon_arch import GriffonMetaModel, GriffonMetaForCausalLM
from griffon.coor_utils import generalized_box_iou


class GriffonLlama2Config(LlamaConfig):
    model_type = "llama2_Griffon"

@dataclass
class CausalLMOutputWithPastCoor(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    coor_output: Optional[torch.FloatTensor] = None

@dataclass
class GreedySearchDecoderOnlyOutputWithLoc(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    locations: Optional[Tuple[torch.FloatTensor]] = None

GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput, GreedySearchDecoderOnlyOutputWithLoc]

class GriffonLlama2Model(GriffonMetaModel, LlamaModel):
    config_class = GriffonLlama2Config

    def __init__(self, config: GriffonLlama2Config):
        super(GriffonLlama2Model, self).__init__(config)


class GriffonLlama2ForCausalLM(LlamaForCausalLM, GriffonMetaForCausalLM):
    config_class = GriffonLlama2Config

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = GriffonLlama2Model(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # self.num_head = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size//2),
        #                               nn.GELU(),
        #                               nn.Linear(config.hidden_size//2, 1))
        # if self.model.config.single_num:
        #     self.num_head = nn.Linear(config.hidden_size, 1)
        #     print("Finish initialize number decoder head")
        # else:
        #     self.num_head = None
        self.num_head = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        numbers: Optional[List[List]] = None,
        regions: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        #import pdb; pdb.set_trace()
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                numbers,
                numbers_mask
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                numbers,
                regions
            )

        # out = super().forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict
        # )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )
        
        hidden_states = outputs[0]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        if self.num_head is not None:
            coors = self.num_head(hidden_states).float()
        else:
            coors = None
        logits = logits.float()

        loss = None
        loss_c = None
        loss_t = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        def giou_box_loss(src_box, tgt_box):
            mask = (src_box[:, 2:] >= src_box[:, :2]).all(-1)
            loss_giou = 1 - torch.diag(generalized_box_iou(
                                        src_box[mask],
                                        tgt_box[mask]))
            return loss_giou.mean()
        
        if numbers is not None and labels is not None:
            shift_coors = coors[numbers_mask].contiguous()
            coor_gt = numbers.contiguous()
            loss_mse = MSELoss()
            shift_coors = shift_coors.view(-1)
            coor_gt = coor_gt.view(-1).to(shift_coors.device)
            loss_c = loss_mse(shift_coors, coor_gt) #*2 + giou_box_loss(shift_coors.view(-1,4), coor_gt.view(-1,4))/5
        
        # if loss_t is not None or loss_c is not None:
        #     loss = loss_c + loss_t
        # print("loss_text: {}, loss_coor: {}".format(loss_t, loss_c))
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        # if isinstance(outputs.hidden_states, tuple):
        #     hid_tuple = outputs.hidden_states + (coors,)
        # else:
        #     hid_tuple = (outputs.hidden_states, coors,)
        if coors is None:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                )
        else:
            return CausalLMOutputWithPastCoor(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                coor_output=coors
            )
        

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        numbers = kwargs.pop("coor_output", None)
        regions = kwargs.pop("regions", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if numbers is not None:
            _inputs["numbers"] = numbers
        if regions is not None:
            _inputs["regions"] = regions
        return _inputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        regions = kwargs.pop("regions", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _, _, _) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, regions=regions)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)


AutoConfig.register("llama2_Griffon", GriffonLlama2Config)
AutoModelForCausalLM.register(GriffonLlama2Config, GriffonLlama2ForCausalLM)
