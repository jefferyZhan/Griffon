![](./docs/logo.jpg)

<div align="center">

# Welcome to Griffon

</div>

Welcome to the official repository of the Griffon Series — including Griffon v1, v2, G, R, and the Vision-R1 reinforcement learning framework. **Griffon begins with fine-grained perception and localization, achieving state-of-the-art performance in visual grounding and referring expression comprehension (REC) — rivaling expert-level object detection models. Beyond its visual strengths, Griffon also demonstrates impressive general-purpose question answering and the ability to identify relevant regions based on a given question to perform reasoning.** Griffon is continuously evolving to tackle increasingly complex vision-language tasks. We are actively maintaining and open-sourcing our progress. Feel free to follow the project and open an issue if you have questions or feedback!

---
***Understand, Think, and Answer: Advancing Visual Reasoning with Large Multimodal Models***

[`📕Paper`](https://arxiv.org/abs/2505.20753) [`🌀Usage`](./Griffon-R/README.md) 
<!-- [`🤗Model`](https://huggingface.co/collections/JefferyZhan/vision-r1-67e166f8b6a9ec3f6a664262) [`🤗Data`](https://huggingface.co/datasets/JefferyZhan/Vision-R1-Data) -->

***Vision-R1: Evolving Human-Free Alignment in Large Vision-Language Models via Vision-Guided Reinforcement Learning***

[`📕Paper`](https://arxiv.org/abs/2503.18013) [`🌀Usage`](./Vision-R1/README.md) [`🤗Model`](https://huggingface.co/collections/JefferyZhan/vision-r1-67e166f8b6a9ec3f6a664262) [`🤗Data`](https://huggingface.co/datasets/JefferyZhan/Vision-R1-Data)

***Griffon-G: Bridging Vision-Language and Vision-Centric Tasks via Large Multimodal Models***

[`📕Paper`](https://arxiv.org/abs/2410.16163) [`🌀Usage`](./README.md) [`🤗Model`](https://huggingface.co/collections/JefferyZhan/griffon-g-6729d8d65cd58b3f40e87794) [`🤗Data🔥`](https://huggingface.co/datasets/JefferyZhan/Griffon-G-CCMD-8M)

***Griffon v2: Advancing Multimodal Perception with High-Resolution Scaling and Visual-Language Co-Referring (ICCV 2025)***

[`📕Paper`](https://arxiv.org/abs/2403.09333) [`🌀Intro`](./docs/README_v2.md) [`🤗Data🔥`](https://huggingface.co/datasets/JefferyZhan/Griffon-V2-Data)

***Griffon: Spelling out All Object Locations at Any Granuality with Large Language Model (ECCV 2024)***

[`📕Paper`](https://arxiv.org/abs/2311.14552) [`🌀Usage`](./docs/README_v1.md) [`🤗Model`](https://huggingface.co/JefferyZhan/Griffon/tree/main)


## Release
- [x] **`2025.08.12`** 🔥🔥**We have release the data of [Griffon v2](https://huggingface.co/datasets/JefferyZhan/Griffon-V2-Data) and [Griffon-G](https://huggingface.co/datasets/JefferyZhan/Griffon-G-CCMD-8M) in the 🤗HuggingFace and also update the [training codes](./docs/TRAIN_README.md).**
- [x] **`2025.08.11`** 🔥🔥**We are glad to annouce that Griffon v2 has been accepted to ICCV 2025**
- [x] **`2025.05.27`** We have released Griffon-R in the [arxiv](https://arxiv.org/abs/2505.20753).
- [x] **`2025.03.25`** We release the Vision-R1 paper, evaluation codes, models, and data. Check out in the [repo](Vision-R1/README.md).
- [x] **`2025.01.15`** Release the evaluation scripts supporting distributed inference.
- [x] **`2024.11.26`** We are glad to release inference code and the model of Griffon-G in [`🤗Griffon-G`](https://huggingface.co/collections/JefferyZhan/griffon-g-6729d8d65cd58b3f40e87794). Training codes will be released later.
- [x] **`2024.07.01`** **Griffon has been accepted to ECCV 2024. Data is released in [`🤗HuggingFace`](https://huggingface.co/datasets/JefferyZhan/Language-prompted-Localization-Dataset)**
- [x] **`2024.03.11`** We are excited to announce the arrival of Griffon v2. Griffion v2 brings fine-grained perception performance to new heights with high-resolution expert-level detection and counting and supports visual-language co-referring. Take a look at our demo first. Paper is preprinted in [`📕Arxiv`](https://arxiv.org/abs/2403.09333).
- [x] **`2023.12.06`** Release the Griffon v1 inference code and model in [`🤗HuggingFace`](https://huggingface.co/JefferyZhan/Griffon/tree/main).
- [x] **`2023.11.29`** Griffon v1 Paper has been released in [`📕Arxiv`](https://arxiv.org/abs/2311.14552).

## What can Griffon do now?
Griffon-G demonstrates advanced performance across multimodal benchmarks, general VQAs, and text-rich VQAs, achieving new state-of-the-art results in REC and object detection.
 **More quantitative evaluation results can be found in our paper.**
![](./docs/griffon-g.jpg)

## Get Started

### 1.Clone & Install

```shell
git clone git@github.com:jefferyZhan/Griffon.git
cd Griffon
pip install -e .
```
Tips: If you encounter any errors while installing the packages, you can always download the corresponding source files (*.whl), which have been verified by us.

---

### 2.Download the Griffon and CLIP models to the checkpoints folder.

| Model                                | Links                                  |
|---------                            |---------------------------------------|
| Griffon-G-9B                        | [`🤗HuggingFace`](https://huggingface.co/JefferyZhan/Griffon-G-gemma2-9B)    |
| Griffon-G-27B                        | [`🤗HuggingFace`](https://huggingface.co/JefferyZhan/Griffon-G-gemma2-27B/tree/main)    |
| clip-vit-large-path14               | [`🤗HuggingFace`](https://huggingface.co/openai/clip-vit-large-patch14)    |
| clip-vit-large-path14-336_to_1022   | [`🤗HuggingFace`](https://huggingface.co/JefferyZhan/clip-vit-large-path14-336_to_1022/tree/main)    |
---

### 3. Training
Please refer to the [Training README](./docs/TRAIN_README.md).

---
### 4.Inference

```shell
# 4.1 Modify the instruction in the run_inference.sh.

# 4.2.1 DO NOT USE Visual Prompt
bash run_inference.sh [CUDA_ID] [CHECKPOINTS_PATH] [IMAGE_PATH]

# 4.2.2 USE Visual Prompt for COUNTING: Input both query image and prompt image splited with comma and specify <region> placeholder in the instruction
bash run_inference.sh [CUDA_ID] [CHECKPOINTS_PATH] [IMAGE_PATH,PROMPT_PATH]
```
Notice: Please pay attention to the singular and plural expressions of objects.

---
### 5.Evaluation

**5.1 Multimodal Benchmark Evaluation**

Please Refer to LLaVA Evaluation or Use VLMEvalKit.


**5.2 COCO Detection Evaluation**


```shell
# Single Node
torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12457 -m griffon.eval.eval_detection --model-path PATH/TO/MODEL --image-folder PATH/TO/coco2017/val2017 --dataset PATH/TO/instances_val2017.json

# Multiple Node
## NODE 0
torchrun --nproc_per_node 8 --nnodes N --node_rank 0 --master_addr MASTER_ADDR --master_port MASTER_PORT -m griffon.eval.eval_detection --model-path PATH/TO/MODEL --image-folder PATH/TO/coco2017/val2017 --dataset PATH/TO/instances_val2017.json --init tcp://MASTER_ADDR:MASTER_PORT
## NODE K(1 to N-1)
torchrun --nproc_per_node 8 --nnodes N --node_rank K --master_addr MASTER_ADDR --master_port MASTER_PORT -m griffon.eval.eval_detection --model-path PATH/TO/MODEL --image-folder PATH/TO/coco2017/val2017 --dataset PATH/TO/instances_val2017.json --init tcp://MASTER_ADDR:MASTER_PORT
```


**5.3 REC Evaluation**

Processed RefCOCO annotation set can be downloaded from this [link](https://drive.google.com/file/d/1Yh1l-f-rLSWkAlXUkZiHmK7oUC9NCmGl/view?usp=sharing).

```shell
# Single Node
torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12457 -m griffon.eval.eval_rec --model-path PATH/TO/MODEL --image-folder PATH/TO/COCO/train2014 --dataset PATH/TO/REF_COCO_ANN

# Multiple Node
## NODE 0
torchrun --nproc_per_node 8 --nnodes N --node_rank 0 --master_addr MASTER_ADDR --master_port MASTER_PORT -m griffon.eval.eval_detection --model-path PATH/TO/MODEL --image-folder PATH/TO/COCO/train2014 --dataset PATH/TO/REF_COCO_ANN --init tcp://MASTER_ADDR:MASTER_PORT
## NODE K(1 to N-1)
torchrun --nproc_per_node 8 --nnodes N --node_rank K --master_addr MASTER_ADDR --master_port MASTER_PORT -m griffon.eval.eval_detection --model-path PATH/TO/MODEL --image-folder PATH/TO/COCO/train2014 --dataset PATH/TO/REF_COCO_ANN --init tcp://MASTER_ADDR:MASTER_PORT
```

## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main) provides the base codes and pre-trained models.
- [Shikra](https://github.com/shikras/shikra) provides insight of how to organize datasets and some base processed annotations.
- [Llama](https://github.com/facebookresearch/llama) provides the large language model.
- [Gemma2](https://arxiv.org/abs/2408.00118) provides the large language model.
- [volgachen](https://github.com/volgachen/Awesome-AI-Environment) provides the basic environment setting config.

## Citation
If you find Griffon useful for your research and applications, please cite using this BibTeX:
```bibtex
@inproceedings{zhan2025griffonv1,
  title={Griffon: Spelling out all object locations at any granularity with large language models},
  author={Zhan, Yufei and Zhu, Yousong and Chen, Zhiyang and Yang, Fan and Tang, Ming and Wang, Jinqiao},
  booktitle={European Conference on Computer Vision},
  pages={405--422},
  year={2025},
  organization={Springer}
}

@misc{zhan2024griffonv2,
      title={Griffon v2: Advancing Multimodal Perception with High-Resolution Scaling and Visual-Language Co-Referring}, 
      author={Yufei Zhan and Yousong Zhu and Hongyin Zhao and Fan Yang and Ming Tang and Jinqiao Wang},
      year={2024},
      eprint={2403.09333},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{zhan2024griffon-G,
  title={Griffon-G: Bridging Vision-Language and Vision-Centric Tasks via Large Multimodal Models},
  author={Zhan, Yufei and Zhao, Hongyin and Zhu, Yousong and Yang, Fan and Tang, Ming and Wang, Jinqiao},
  journal={arXiv preprint arXiv:2410.16163},
  year={2024}
}
```

## License

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

The data and checkpoint is licensed for research use only. All of them are also restricted to uses that follow the license agreement of LLaVA, LLaMA, Gemma2, and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.