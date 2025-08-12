# Training

Griffon v2 and Griffon-G share the same training code with data format of Griffon v2 sightly different. As the Griffon-G further advances the Griffon v2, we'd like to introduce the training procedure following the settings of Griffon-G.

## Data

### Downloads

1. Download the [ShareGPT4V](https://github.com/ShareGPT4Omni/ShareGPT4V) caption data and images, following the official instruction.

2. Download the pretrain data and instruction-following data from our huggingface repo [`🤗Data`](https://huggingface.co/datasets/JefferyZhan/Griffon-G-CCMD-8M) and the images following the huggingface repo.

### Data Input
We mainly support two ways of inputting data. For general instruction-following data such as captions and QA, we adopt the LLaVA format, which can be directly input by specifying ```--data-path```. When introducing location-related data, we input it by setting ```--multi_path``` in a form that includes the task type, annotations, and image paths (separated by spaces). An example is as follows.

```
# [Task Type] [Annotation Path] [Image Path]
LAZY_0 & griffon_instruct.json & llava_15_data/instruct
DET_0 & objects365.jsonl & Objects365-2023/full-images
REG_0 & vg_reg.jsonl & VG/VG_ALL
VG_0 & grefcoco.jsonl & train2014
VG_1 & refcoco.jsonl & train2014
```
- Task Type: ```LAZY``` for instruction data, ```DET``` for detection data, ```REG``` for Referring Expression Generation data, ```VG``` for visual grounding and Referring Expression Comprehension data, and ```VISJ``` for counting data.
- For the same task type of data, number each line with "_"
- For the specific data format, please refer to the data we release in the Huggingface.

### Path Customization
After following the huggingface repo to download the annodations, update the annotation path and corresponding image path for each line in ```scripsts/datasets```.

## Train
We release the scripts for each model in each stage. Besides the Gemma2-based and LLAMA2-based models, we also support the leading Qwen2.5-based models. We will release the scripts of LLAMA2 and Qwen2.5 later.

### Stage 1: Modality Alignment Initialization
For the alignment stage, set the ```--tune_mm_mlp_adapter True``` and ```--version plain```.

```
bash scripts/*/alignment.sh
```

### Stage 2: Paradigm Pre-Adaptation Pre-training
Modify the data paths in the ```scripts/datasets/pretrain.txt``` according to your customized paths, and run the command.
```
bash scrips/*/pretrain.sh
```

### Stage 3: Comprehensive Instruction Tuning
Modify the data paths in the ```scripts/datasets/pretrain.txt``` according to your customized paths, and run the command. In this version, we recommend to split the training of visual tokenizer and the other modules, which we finds more stable and higher-performance. You can also follow the training design of Griffon v2 by modify the data inputs.
```
bash scrips/*/sft.sh
bash scrips/*/visual_referring.sh
```

If you want to use multiple nodes for training, modify the training scripts as follows and also the hostfile.

```
deepspeed --hostfile=scripts/hostfile.txt griffon/train/train_mem.py 
```

```
gpu-compute01 slots=8
gpu-compute02 slots=8
```

## Costomized Fine-tuning for Your Own Scenario

We support further fine-tunes our model for your own scenario. Please first modify you data following the format as we provided. Fill in the data setting in the ```scripts/datasets/custom.txt```. For different model, you should modify the ```--version``` accordingly, and can use the ```sft.sh``` script for training. To control the trainable parts, we provide the following parameters:
- ```freeze_llm```: Default to False, and set to True for freezing LLM
- ```freeze_mm_mlp_adapter```: Default to False, and set to True for freezing adapter
- ```freeze_vision_tower```: Default to False, and set to True for freezing vision tower