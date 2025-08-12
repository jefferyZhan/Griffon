import re
import torch
from collections import defaultdict
from griffon.utils import auto_rank0_print

def merge_nums(names):
    pattern = re.compile(r'(\D+)(\d+)(\D.*)')
    groups = defaultdict(list)
    for name in names:
        m = pattern.match(name)
        if m:
            key = m.group(1) + '{}' + m.group(3)
            num = int(m.group(2))
            groups[key].append(num)
        else:
            groups[name].append(None)

    merged = []
    for key, nums in groups.items():
        if nums[0] is None:  
            merged.append(key)
        else:
            nums.sort()
            ranges = []
            start = end = nums[0]
            for n in nums[1:]:
                if n == end + 1:
                    end = n
                else:
                    if start == end:
                        ranges.append(str(start))
                    else:
                        ranges.append(f"{start}-{end}")
                    start = end = n
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            merged.append(key.format('[' + ','.join(ranges) + ']'))
    return merged

def print_trainable_params_table(model):
    params = [(name, p.shape) for name, p in model.named_parameters() if p.requires_grad]

    param_names = [name for name, _ in params]
    merged_names = merge_nums(param_names)

    auto_rank0_print("="*70)
    auto_rank0_print("{:<45} {:<20}".format("Parameter Name", "Shape"))
    auto_rank0_print("-"*70)
    used = set()
    for merged_name in merged_names:
        matches = []
        if '{' in merged_name:
            # i.e. layer.[1-3].weight -> layer.1.weight, layer.2.weight, layer.3.weight
            template = re.sub(r'\[\S+\]', r'(\d+)', merged_name.replace('{','').replace('}',''))
            regex = re.compile(template)
            matches = [(name, shape) for name, shape in params if regex.match(name)]
        else:
            matches = [(name, shape) for name, shape in params if name == merged_name]
        if merged_name not in used:
            shapes = set([str(shape) for _, shape in matches])
            auto_rank0_print("{:<45} {:<20}".format(merged_name, ', '.join(shapes)))
            used.add(merged_name)
    auto_rank0_print("="*70)