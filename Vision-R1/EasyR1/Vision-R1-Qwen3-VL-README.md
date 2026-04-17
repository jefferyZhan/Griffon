# 🚀 Vision-R1: Qwen3-VL Training & Evaluation

[![Back to Main](https://img.shields.io/badge/Return_to-Vision--R1_Main-blue.svg?style=for-the-badge)](../README.md) <!-- 请根据实际路径修改链接 -->

This document provides a comprehensive guide for training and evaluating **Qwen3-VL** using **Vision-R1**, powered by the [EasyR1](https://github.com/hiyouga/EasyR1) framework. 

We provide an end-to-end pipeline from environment setup and model training to checkpoint merging and standardized evaluation.

---

## 🛠️ Installation

Clone the EasyR1 repository and install it in editable mode:

```bash
cd Vision-R1/EasyR1
pip install -e .
```

---

## 🏃‍♂️ Training Qwen3-VL with Vision-R1

Start the training process using the provided configuration file:

```bash
python3 -m verl.trainer.main config=./examples/qwen3_vl_8b_vision_r1.yaml
```

> **💡 Note for Qwen2.5-VL Users:**
> Due to differences in grounding prompts and coordinate representation methods between Qwen3-VL and Qwen2.5-VL, if you wish to train the Qwen2.5-VL model using this codebase, you must adjust the dataset loading logic in `verl/utils/dataset.py` and the corresponding settings in `visionr1.py` or use our Open-R1 version.

### Merge Checkpoint
After training, convert the deepspeed/actor checkpoints into the standard Hugging Face format:

```bash
python3 scripts/model_merger.py --local_dir ./checkpoints/Qwen3-VL-8B-Instruct-Vision-R1/global_step_200/actor
```

---

## 📊 Evaluation

Compared to the official implementation, our evaluation pipeline offers several key enhancements designed for out-of-the-box usability:
- **Automated Data Preparation:** Provided scripts for seamless ODINW-13 and COCO 2017 downloads.
- **Clear Configurations:** Explicit instructions on structuring `dataset_config.py`.
- **Environment Compatibility:** Set Qwen3-VL coordinate representation formats explicitly via OS environment variables.
- **COCO Benchmark Support:** Fully integrated COCO evaluation metrics.

### 1. Download Datasets

**Download ODINW-13:**
```bash
cd Vision-R1/eval/eval_qwen3_vl
python download_odinw.py --dataset_path ./odinw
```

**Download COCO 2017:**
```bash
# Download, extract, and clean up annotations
mkdir -p coco/coco2017 
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip 
unzip annotations_trainval2017.zip -d coco/coco2017 
rm annotations_trainval2017.zip

# Download, extract, and clean up val images
wget http://images.cocodataset.org/zips/val2017.zip 
unzip val2017.zip -d coco/coco2017 
rm val2017.zip
```

<details>
<summary><b>📂 Click to view the expected directory structure</b></summary>

```text
/path/to/odinw/
├── dataset_config.py          # Dataset configuration file (required)
├── AerialMaritimeDrone/       # Individual datasets
│   ├── large/
│   │   ├── train/
│   │   └── test/
│   └── tiled/
├── Aquarium/
├── Cottontail Rabbits/
├── EgoHands/
├── NorthAmerica Mushrooms/
├── Packages/
├── Pascal VOC/
├── Pistols/
├── Pothole/
├── Raccoon/
├── ShellfishOpenImages/
├── Thermal Dogs and People/
└── Vehicles OpenImages/

/path/to/coco/
├── dataset_config.py          # Dataset configuration file (required)
├── coco2017/
│   ├── annotations
│   │   ├── instances_val2017.json
│   │   └── ...
│   └── val2017
```
</details>

### 2. Install Evaluation Dependencies
```bash
pip install -r requirements.txt
```

### 3. Inference & Evaluation

Before running the scripts, please update the `model-path`, `data-dir`, `output-file`, and `input-file` arguments inside the `.sh` files according to your local setup. For other inference settings, you can update them based on your device.

```bash
bash infer_instruct.sh
bash eval_instruct.sh
```
> **⚠️ Important:** Currently, only the **Instruct** evaluation mode has been fully verified.

---

## 🏆 Experimental Results

Our proposed **Vision-R1** still demonstrates significant improvements across various object detection benchmarks, particularly enhancing the performance of Qwen3-VL-8B. These results further verify the generalization capabilites of our **Vision-R1**.

### COCO2017 Object Detection Results

| Model | mAP | mAP@50 | mAP@75 | mAP (S) | mAP (M) | mAP (L) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-7B | 0.177 | 0.273 | 0.180 | — | — | — |
| Qwen3-VL-8B | 0.307 | 0.455 | 0.317 | 0.149 | 0.348 | 0.543 |
| **Qwen3-VL-8B + Vision-R1** | **0.366** | **0.526** | **0.391** | **0.178** | **0.418** | **0.571** |

### 📊 ODINW-13 mAP Comparison

| Model | Aerial | Aquarium | Rabbits | EgoHands | Mushrooms | Packages | Pascal | Pistols | Pothole | Raccoon | Shellfish | Thermal | Vehicles | **Avg.** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-7B | 0.078 | 0.203 | **0.735** | 0.322 | 0.070 | **0.576** | 0.498 | 0.485 | 0.074 | 0.401 | **0.427** | 0.380 | 0.563 | 0.370 |
| Qwen3-VL-8B | **0.085** | 0.275 | 0.659 | 0.507 | 0.653 | 0.438 | 0.464 | 0.559 | 0.212 | **0.602** | 0.305 | 0.562 | 0.555 | 0.452 |
| **Qwen3-VL-8B + Vision-R1** | 0.027 | **0.298** | 0.647 | **0.591** | **0.785** | 0.438 | **0.513** | **0.561** | **0.245** | 0.596 | 0.380 | **0.633** | **0.597** | **0.485** |

*(Note: Best results in the ODINW-13 evaluation are highlighted in **bold**.)*

---

## 📑 Citation

If you find our repository or paper useful, please star this repo and cite our work:

```bibtex
@misc{zhan2025visionr1evolvinghumanfreealignment,
      title={Vision-R1: Evolving Human-Free Alignment in Large Vision-Language Models via Vision-Guided Reinforcement Learning}, 
      author={Yufei Zhan and Yousong Zhu and Shurong Zheng and Hongyin Zhao and Fan Yang and Ming Tang and Jinqiao Wang},
      year={2025},
      eprint={2503.18013},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.18013}, 
}
```

> **⚠️ Note on Datasets:**
> This repository provides scripts to download evaluation datasets for your convenience. If you use the **COCO 2017** or **ODINW** datasets in your research, please ensure you also cite their respective original papers and authors.
