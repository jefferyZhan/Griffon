import numpy as np
import json

from torch.utils.data import Dataset, ConcatDataset, Subset
from functools import partial

from .base_class import DataArguments, LazySupervisedDataset

from .det import DETDataset, RANDDETDataset
from .reg import REGDataset
from .vg import VGDataset
from .vvn import VisualDetDataset_JSONL_v2
import transformers

seed=48
def process_dataset(txt_file):
    # The format is task&data_path&image_folder
    data = open(txt_file,"r").readlines()
    data_dict = {}
    for item in data:
        item = item.strip()
        dtype, data_path, image_folder = item.split("&")
        data_dict[dtype.strip()] = (data_path.strip(), image_folder.strip())
    return data_dict

class MultiConcatDataset(Dataset):
    _repr_indent = 4

    def __init__(self, multi_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 debug=False
                 ):
        #dataset_dict: {"REC": (image_folder, data_path)}
        assert data_args.data_path == None and data_args.image_folder == None, "If use multi paths, please do not input single path and folder."
        dataset_dict = process_dataset(multi_path)
        self.dataset_dict = dataset_dict

        det_template = "griffon/datasets/templates/nvn.json"
        vis_template = "griffon/datasets/templates/visual_search.json"
        reg_template = "griffon/datasets/templates/REG.json"
        ovn_template = "griffon/datasets/templates/1vn.json"
        vg_template = "griffon/datasets/templates/1vn.json"

        detdataset = partial(DETDataset, tokenizer=tokenizer, data_args=data_args, template_file=det_template, debug=debug)
        randdetdataset = partial(RANDDETDataset, tokenizer=tokenizer, data_args=data_args, template_file=vg_template, debug=debug)
        regdataset = partial(REGDataset, tokenizer=tokenizer, data_args=data_args, template_file=reg_template, debug=debug)
        vgdataset = partial(VGDataset, tokenizer=tokenizer, data_args=data_args, template_file=vg_template, debug=debug)
        lazydataset = partial(LazySupervisedDataset, tokenizer=tokenizer, data_args=data_args)
        countdataset = partial(VisualDetDataset_JSONL_v2, tokenizer=tokenizer, data_args=data_args, unshared_visual_encoder=True, template_file=vis_template, debug=debug)
        
        self.map_func ={
            # "REC": recdataset,
            "DET": detdataset,
            "LAZY": lazydataset,
            "REG": regdataset,
            "VG": vgdataset,
            "VISJ": countdataset,
            "RANDDET": randdetdataset
        }
        datasets = []
        self.modality_lengths = []
        self.lengths = []
        self.task_lengths = []
        for tp, (data_path, image_folder) in self.dataset_dict.items():
            if tp == "DET_oi" or tp == "DET_obj":
                temp = DETDataset(data_path=data_path, image_folder=image_folder, tokenizer=tokenizer, data_args=data_args, template_file=det_template, debug=debug, cate_sample=True)
                # print(f"{tp} & {data_path}:")
                # print(temp[1])
            else:
                temp = self.map_func[tp.split("_")[0]](data_path=data_path, image_folder=image_folder)
                # print(f"{tp} & {data_path[-1]}:")
                # print(temp[1])
            datasets.append(temp)
        for dataset in datasets:
            self.modality_lengths.extend(dataset.modality_lengths)
            self.lengths.extend(dataset.lengths)
            self.task_lengths.extend(dataset.task_lengths)
        self.concat_dataset = ConcatDataset(datasets)    

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, index):
        return self.concat_dataset[index]

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [
            f"Number of datapoints: {self.__len__()}",
        ]
        for i, ds in enumerate(self.concat_dataset.datasets):
            body.append(f"Subset {i + 1}/{len(self.concat_dataset.datasets)}")
            body += ds.__repr__().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)


class MultiBalancedDataset(Dataset):
    """
        Build on MultiConcatDataset to support the combined sampling for VVN
        multi_multi_path: path to store data in different type
    """
    _repr_indent = 4

    def __init__(self, multi_multi_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 debug=False
                 ):
        paths = open(multi_multi_path, "r").readlines()
        datasets = []
        datasets_length = []
        datasets_modal_length = []
        datasets_record = []
        for item in paths:
            scale, path = item.split("&")
            scale = float(scale.strip())
            path = path.strip()
            dataset = MultiConcatDataset(path, tokenizer, data_args)
            if scale == 1:
                datasets.append(dataset)
                datasets_length.extend(dataset.lengths)
                datasets_modal_length.extend(dataset.modality_lengths)
                datasets_record.append(len(dataset))
            else:
                target_len = int(len(dataset) * scale)
                #default to random sample
                rng = np.random.default_rng(seed)
                indices = list(range(len(dataset)))
                rng.shuffle(indices)
                indices = indices[:target_len]
                #获取两类length
                modal_length = np.array(dataset.modality_lengths)
                length = np.array(dataset.lengths)
                maped_modal_length = modal_length[indices].tolist()
                maped_length = length[indices].tolist()
                dataset = Subset(dataset, indices)
                datasets.append(dataset)
                datasets_length.extend(maped_length)
                datasets_modal_length.extend(maped_modal_length)
                datasets_record.append(len(dataset))
        self.concat_dataset = ConcatDataset(datasets)
        self.lengths = datasets_length
        self.modality_lengths = datasets_modal_length
        self.dataset_recoder = datasets_record

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, index):
        return self.concat_dataset[index]

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [
            f"Number of datapoints: {self.__len__()}",
        ]
        for i, ds in enumerate(self.concat_dataset.datasets):
            body.append(f"Subset {i + 1}/{len(self.concat_dataset.datasets)}")
            body += ds.__repr__().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)