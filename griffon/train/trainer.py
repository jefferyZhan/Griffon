from asyncio import FastChildWatcher
import os
import torch
import re

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # if len(additional_batch) > 0:
    #     megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]

#####################################################################
############### Tools For Different Batch Sampler ###################
#####################################################################
def get_task_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # 此函数为了让定位数据、通用数据及纯文本数据分离在不同的batch中
    # 将length组成为一个个元组，每一个元组（task_id, length),task_id为任务类型，length为该条数据的长度
    # id=0: pure text, id=1: general task, id=2: loc task, id=3: det task
    assert all(l != 0 for (id, l) in lengths), "ALL DATA SHOULD NOT HAVE ZERO LENGTH."
    if all(id == 0 for (id, l) in lengths) or all(id == 1 for (id, l) in lengths) or all(id == 2 for (id, l) in lengths):
        # all samples are in the same task
        batch_lengths = [lengths[i][1] for i in range(len(lengths))]
        return get_length_grouped_indices(batch_lengths, batch_size, world_size, generator=generator) 

    general_indices, general_lengths = zip(*[(i, l) for i, (id, l) in enumerate(lengths) if id==1])
    lang_indices, lang_lengths = zip(*[(i, l) for i, (id, l) in enumerate(lengths) if id==0])
    loc_indices, loc_lengths = zip(*[(i, l) for i, (id, l) in enumerate(lengths) if id==2])
    #det_indices, det_lengths = zip(*[(i, l) for i, (id, l) in enumerate(lengths) if id==3])

    general_shuffle = [general_indices[i] for i in get_length_grouped_indices(general_lengths, batch_size, world_size, generator=None)]
    loc_shuffle = [loc_indices[i] for i in get_length_grouped_indices(loc_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    #det_shuffle = [det_indices[i] for i in get_length_grouped_indices(det_lengths, batch_size, world_size, generator=None)]

    megabatch_size = world_size * batch_size
    general_megabatches = [general_shuffle[i : i + megabatch_size] for i in range(0, len(general_shuffle), megabatch_size)]
    loc_megabatches = [loc_shuffle[i : i + megabatch_size] for i in range(0, len(loc_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]
    #det_megabatches = [det_shuffle[i : i + megabatch_size] for i in range(0, len(det_shuffle), megabatch_size)]

    last_general = general_megabatches[-1]
    last_lang = lang_megabatches[-1]
    last_loc = loc_megabatches[-1]
    #last_det = det_megabatches[-1]
    
    # We still drop the last one
    ############### general and localization iterleaved #########################
    # additional_batch = last_general + last_lang + last_loc + last_det
    # megabatches = general_megabatches[:-1] + lang_megabatches[:-1] 
    # #additional_batch = last_general  + last_loc
    # #megabatches = general_megabatches[:-1] + loc_megabatches[:-1]
    # megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    # det_megabatches = det_megabatches[:-1] + loc_megabatches[:-1]
    # det_batch_indices = torch.randperm(len(det_megabatches), generator=generator)
    
    # megabatches_final = []
    # for id in range(max(len(megabatch_indices), len(det_batch_indices))):
    #     if id < min(len(megabatch_indices), len(det_batch_indices)):
    #         megabatches_final.append(det_megabatches[det_batch_indices[id]])
    #         megabatches_final.append(megabatches[megabatch_indices[id]])
    #     else:
    #         megabatches_final.append(megabatches[megabatch_indices[id]])
    # print("############ Begin to task split dataloader #######################")
    # print(f"total {len(megabatches_final)} batches: {len(det_megabatches)} for det, {len(megabatches)} for general.")
    ############### #################################### #########################
    
    ######################### DIFFERENT BATCH RANDOM SAMPLE ######################
    additional_batch = last_general + last_lang + last_loc 
    megabatches = general_megabatches[:-1] + lang_megabatches[:-1] + loc_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]
    print(f"total {len(megabatches)} batches")
    #################################################################################

    # if len(additional_batch) > 0:
    #     megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]

def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
        group_by_task: bool = False
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality
        self.group_by_task = group_by_task

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_task:
            indices = get_task_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        elif self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

######################################################################
############### Tools For Layer Wise Learning Rate Decay #############
#####################################################################
class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_transformer(var_name, len(self.values))

def get_num_layer_for_transformer(param_name, num_max_layer):
    layer_0 = {
        "patch_embed", 
        "pos_embed", 
        "cls_token", 
        "mask_token", 
        "conv1",
        "positional_embedding",
        "token_embedding",
        "transformer.embeddings.word_embeddings",
        "transformer.embeddings.position_embeddings",
        "transformer.embeddings.token_type_embeddings",
    }

    if any(l in param_name for l in layer_0):
        return 0

    block_regex = re.compile(r"blocks\.([0-9]+)\.")
    match_block = block_regex.search(param_name)

    #huggingface->text.transformer.encoder.layer
    layer_regex = re.compile(r"layer\.([0-9]+)\.") 
    match_layer = layer_regex.search(param_name)
    if match_block is not None:
        return int(match_block.group(1)) + 1
    elif match_layer is not None:
        return int(match_layer.group(1)) + 1
    else:
        return num_max_layer - 1

SKIP = []

#######################################################
####################### END ###########################
#######################################################

class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        
        if self.args.group_by_task_length:
            lengths = self.train_dataset.task_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=False,
                group_by_task=True
            )
        elif self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                if self.args.mm_vision_tower_lr:
                    # ZYF Modify
                    # Modify to support layer wise finetuning
                    
                    ########################## Get Visual Parameters #######################
                    vision_tower_parameters = [name for name, _ in opt_model.named_parameters() if "vision_tower" in name]
                    
                    ########################## SAME Different Rate #########################
                    print(f"Set the learning rate of vision tower to {self.args.mm_vision_tower_lr}.")
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in vision_tower_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in vision_tower_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in vision_tower_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_vision_tower_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in vision_tower_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_vision_tower_lr,
                        },
                    ]
                    #######################################################################

                    ########################## Layer Wise Setting ##########################
                    print(f"Set the learning rate of vision tower to {self.args.mm_vision_tower_lr} and lr decay to {self.args.mm_vision_tower_lr_decay} and weight decay to {self.args.mm_vision_tower_wd_decay}")
                    # visual_num_layers = 24 # For ViT_L usually
                    # assigner = LayerDecayValueAssigner(list(self.args.mm_vision_tower_lr_decay ** (visual_num_layers + 1 - i) for i in range(visual_num_layers + 2)))
                    # get_layer_scale = assigner.get_scale
                    # get_layer_num = assigner.get_layer_id
                    # parameter_group_names = {}
                    # parameter_group_vars = {}
                    # for name, param in opt_model.named_parameters():
                    #     if not param.requires_grad:
                    #         continue
                        
                    #     if name not in vision_tower_parameters:
                    #         group_name = "Non_vision_tower"
                    #         if name in decay_parameters:
                    #             group_name = "%s_decay" % group_name
                    #             this_weight_decay = self.args.weight_decay
                    #         else:
                    #             group_name = "%s_no_decay" % group_name
                    #             this_weight_decay = 0.0
                    #         layer_id = None
                    #     else:
                    #         if param.ndim <= 1 and name.endswith(".bias"):
                    #             group_name = "no_decay"
                    #             this_weight_decay = 0.0
                    #         else:
                    #             group_name = "decay"
                    #             this_weight_decay = self.args.mm_vision_tower_wd_decay
                    #         layer_id = get_layer_num(name)
                    #         group_name = "layer_%d_%s" % (layer_id, group_name)
                        
                    #     if group_name not in parameter_group_names:
                    #         if layer_id is None:
                    #             parameter_group_names[group_name] = {
                    #                 "weight_decay": this_weight_decay,
                    #                 "params": [],
                    #             }
                    #             parameter_group_vars[group_name] = {
                    #                 "weight_decay": this_weight_decay,
                    #                 "params": [],
                    #             }
                    #         else:
                    #             scale = get_layer_scale(layer_id)
                    #             parameter_group_names[group_name] = {
                    #                 "weight_decay": this_weight_decay,
                    #                 "params": [],
                    #                 "lr_scale": scale,
                    #                 "lr": self.args.mm_vision_tower_lr
                    #             }
                    #             parameter_group_vars[group_name] = {
                    #                 "weight_decay": this_weight_decay,
                    #                 "params": [],
                    #                 "lr_scale": scale,
                    #                 "lr": self.args.mm_vision_tower_lr
                    #             }
                    #     parameter_group_vars[group_name]["params"].append(param)
                    #     parameter_group_names[group_name]["params"].append(name)
                    
                    # optimizer_grouped_parameters = list(parameter_group_vars.values())
                    ########################## Layer Wise Setting ##########################
                
                else:
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                        },
                    ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            #     self.optimizer = OSS(
            #         params=optimizer_grouped_parameters,
            #         optim=optimizer_cls,
            #         **optimizer_kwargs,
            #     )
            # else:
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            try:
                super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)
            except:
                super(LLaVATrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
