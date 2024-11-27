import os
os.environ['MEMORY_FS_ROOT'] = "/storage-root/datasets" 
os.environ['JOBLIB_TEMP_FOLDER']  = "/storage-root/datasets"
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import transformers
import re

from griffon.constants import IGNORE_INDEX
from griffon.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from griffon.conversation import conv_templates, SeparatorStyle
from griffon.model.builder import load_pretrained_model
from griffon.utils import disable_torch_init
from griffon.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import itertools
from functools import partial

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return {"input_ids": input_ids, "image_tensor": image_tensor, "info": line}

    def __len__(self):
        return len(self.questions)

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

def collate_fn(batches):
    # dataloader会将数据自动转为cuda
    input_ids = []
    image_tensors = []
    infos = []
    for _ in batches:
        image_tensors.append(_["image_tensor"])
        input_ids.append(_["input_ids"])
        infos.append(_["info"])
    if len(batches) > 1:
        input_ids = padding_left(input_ids)
        return input_ids, torch.cat(image_tensors, dim=0), infos
    else:
        return input_ids[0].unsqueeze(0), image_tensors[0], infos[0]
    
# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=1):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, sampler=InferenceSampler(len(dataset)), batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, collate_fn=partial(collate_fn))
    return data_loader

def eval_model(args):
    # Model
    # disable_torch_init()
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
        init_method=str("tcp://127.0.0.1:20058")
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if "llama" in model_name.lower():
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, torch_dtype=torch_dtype, device_map="cuda")

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    conv = conv_templates[args.conv_mode]
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str, "\n###"]

    eval_outputs = []
    with torch.inference_mode():
        for i, (input_ids, image_tensors, infos) in tqdm(enumerate(data_loader)):

            idx = infos["question_id"]
            cur_prompt = infos["text"]

            input_ids = input_ids.to(device='cuda', non_blocking=True)
            
            if "qwen" in args.model_path.lower() or "gemma" in args.model_path.lower():
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask = None
            
            if "gemma" in args.model_path.lower():
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            else:
                stopping_criteria = None
            #import pdb; pdb.set_trace()

            
            if stopping_criteria is not None:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensors.unsqueeze(0).to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
                    attention_mask=attention_mask,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
            else:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensors.to(dtype=torch_dtype, device='cuda', non_blocking=True),
                    attention_mask=attention_mask,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            if "qwen" in args.model_path.lower() or "gemma" in args.model_path.lower():
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                # outputs = outputs.strip().replace(": ","").replace("\n###", "")
                # import pdb; pdb.set_trace()
            else:
                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
            # if i == 0:
            #     print(outputs)

            ans_id = shortuuid.uuid()
            if "label" in infos:
                eval_outputs.append(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "label": infos["label"],
                                    # "raw_text": infos["raw_text"],
                                    "category": infos["category"],
                                    "metadata": {}}) + "\n")
            else:
                eval_outputs.append(json.dumps({"question_id": idx,
                                        "prompt": cur_prompt,
                                        "text": outputs,
                                        "answer_id": ans_id,
                                        "model_id": model_name,
                                        "metadata": {}}) + "\n")
        # ans_file.flush()
    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, eval_outputs)
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:
        ans_file.writelines(merged_outputs)
        ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    eval_model(args)
