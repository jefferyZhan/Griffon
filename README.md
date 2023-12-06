![](./logo.jpg)

<div align="center">

# Welcome to Griffon

</div>

This is the offical repo of the paper **Griffon: Spelling out All Object Locations at Any Granuality with Large Language Model**. Griffon is a strong baseline firstly capable of localizing objects based on the free-form texts at any granularity. Surpassing previous localization-related Large Vision Language Models, Griffon can handle all localization scenarios in real world for daily usage. This is the first version of Griffon. We will upgrade Griffon continously, equipping it with more precise capabilities in the localization and other instance-level tasks, e.g. Referring Comprehension Segmentation and Instance Segmentation. Also, corresponding datasets will be improved too. Welcome to follow this repo.

## What can Griffon do?
Griffon can perform localization with free-form text inputs. The command may direct to a single referent, a group of objects represented by a category name or a phase, and plenty of categories. Also, if you are not sure whether an instance appear in the image, Griffon can help. We provide some demo images to show how to use Griffon, and more demos can be found in our paper.
![](./demo.jpg)

## Performance

### REC
| Type        | Model            | RefCOCO val | RefCOCO test-A | RefCOCO test-B | RefCOCO+ val | RefCOCO+ test-A | RefCOCO+ test-B | RefCOCOg val | RefCOCOg test |
|-------------|------------------|-------------|----------------|----------------|--------------|-----------------|-----------------|--------------|---------------|
| Specialists | MDETR            | 87.5        | 90.4           | 82.7           | 81.1         | 85.5            | 73.0            | 83.3         | 83.3          |
|             | TransVG          | 81.0        | 82.7           | 78.4           | 64.8         | 70.7            | 56.9            | 68.7         | 67.7          |
|             | G-DINO-L         | 90.6        | 93.2           | 88.2           | 82.8         | 89.0            | 75.9            | 86.1         | 87.0          |
|             | UNINEXT-L        | 91.4        | 93.7           | 88.9           | 83.1         | 87.9            | 76.2            | 86.9         | 87.5          |
| Generalists | OFA-L            | 80.0        | 83.7           | 76.4           | 68.3         | 76.0            | 61.8            | 67.8         | 67.5          |
|             | Qwen-VL          | 89.4        | 92.3           | 85.3           | 83.1         | 88.3            | 77.2            | 85.6         | 85.5          |
|             | PINK             | 88.7        | 92.1           | 84.0           | 81.8         | 88.2            | 73.9            | 83.9         | 84.3          |
|             | FERRET-13B       | 89.5        | 92.4           | 84.4           | 82.8         | 88.1            | 75.2            | 85.8         | 86.3          |
|             | Shikra-13B       | 87.8        | 90.6           | 80.2           | 82.9         | 87.8            | 74.4            | 82.6         | 83.2          |
|             | **Griffon-13B+** | 89.4        | 92.5           | 84.6           | 83.3         | 88.4            | 76.0            | 85.1         | 86.1          |
|             | **Griffon-13B*** | **90.1**    | **93.4**       | **86.1**       | **84.8**     | **90.5**        | **77.8**        | **86.1**     | **87.2**      |

### Object Detection
| Type        | Model            | Input Size | Epochs | mAP  | AP50 | AP75 | APS | APM  | APL  |
|-------------|------------------|------------|--------|------|------|------|-----|------|------|
| Specialists | FRCNN-R50        | 448        | 1x     | 26.3 | 42.1 | 27.5 | 4.6 | 27.7 | 49.9 |
|             | Pix2Seq-R50      | 1333       | 1x     | 43.0 | 61.0 | 45.6 | 25.1| 46.9 | 59.4 |
|             | DETR-DC5         | 1333       | 1x     | 15.5 | 29.4 | 14.5 | 4.3 | 15.1 | 26.9 |
| Generalists | **Griffon-13B+** | 448        | 1      | 24.8 | 40.6 | 25.1 | 5.9 | 25.5 | 48.7 |

### Phrase Grounding
| Type        | Model           | ANY val | ANY test | MERGED val | MERGED test |
|-------------|-----------------|---------|----------|------------|-------------|
| Specialists | BAN             | -       | 67.9     | -          | -           |
|             | DDPN            | -       | -        | 72.8       | 73.5        |
|             | VisualBert      | 70.4    | 71.3     | -          | -           |
|             | MDETR           | 82.5    | 83.4     | 82.3       | 83.8        |
| Generalists | UniTAB          | -       | -        | 78.8       | 79.6        |
|             | FERRET-13B      | -       | -        | 81.1       | **84.8**    |
|             | Shikra-13B      | -       | -        | 77.4       | 78.4        |
|             | **Griffon-13B+**| **83.7**| **84.2** | **82.0**   | 82.8        |


## TODO List
- [x] **`2023.12.06`** Release the inference code and model.
- [x] **`2023.11.29`** Paper has been released in [`📕Arxiv`](https://arxiv.org/abs/2311.14552).
- [ ] Release the Language-prompted Detection Dataset.
- [ ] Release the online demo.
- [ ] Integrate the segmentation function.
- [ ] Improve performance continously...

## Get Started

First clone this repo.
```shell
git clone git@github.com:jefferyZhan/Griffon.git
```

Install the packages. We recommend to build with an env file and install additional packages. Thanks to [volgachen](https://github.com/volgachen/Awesome-AI-Environment) for the base environment which is a general environment for lots of repos.
```shell
cd Griffon
conda env create -f conda_env.yaml
pip install -r requirements
pip install -e .
```

Download the model weights from the link below and the visual encoder weights from CLIP to the checkpoints folder. Also, resize the position embedding.
| Model    | Link           |
|----------| -------------------------------------------------------------------- |
| Griffon+ | [`🤗HuggingFace`](https://huggingface.co/JefferyZhan/Griffon/tree/main) |
| ViT/L-14@224 | [`💡clip`](https://huggingface.co/openai/clip-vit-large-patch14/tree/main) |

```shell
# resize the clip model to 448 to get the preprocessor
python tools/resize_clip_pos.py --model-path checkpoints/clip-vit-large-patch14 --new-size 448 --patch-size 14 --save-path checkpoints/clip-vit-large-patch14-448
# replace the config and preprocess_config
cp tools/preprocessor_config.json checkpoints/clip-vit-large-patch14-448/
cp tools/config.json checkpoints/clip-vit-large-patch14-448/
```

Inference with command. Due to the distribution of different category and different scenario data at this stage, we **encourage to try different format inputs or add some customed process aftering acquiring the results for desired output.**. We are working now to further increase the scenario diversity and category diversity. A more flexible and general model will be soon released as the next version after the internal evaluation.
```shell
bash demo/demo.sh IMAGE_PATH COMMAND
# Localize Single Referent
bash demo/demo.sh demo/1v1.jpg "Is there a motorcycle on the far left of the photo?"
# Multi Categories with Multi Objects
bash demo/demo.sh demo/nvn.jpg "Examine the image for any objects from the category set. Report the coordinates of each detected object. The category set includes person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush. The output format for each detected object is class-name-[top-left coordinate, bottom-right coordinate] e.g. person-[0.001, 0.345, 0.111, 0.678]. Concatenate them with &."
# One Categories with Multi Objects
bash demo/demo.sh demo/1vn.jpg "In this picture, identify and locate all the people in the front."
```

## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main) provides the base codes and pre-trained models.
- [Shikra](https://github.com/shikras/shikra) provides the insight of how to organize datasets and some base processed annotations.
- [Llama](https://github.com/facebookresearch/llama) provides the large language model.
- [volgachen](https://github.com/volgachen/Awesome-AI-Environment) provides the basic environment setting config.

## License

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

The data and checkpoint is licensed for research use only. All of them are also restricted to uses that follow the license agreement of LLaVA, LLaMA and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.