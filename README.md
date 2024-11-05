![](./logo.jpg)

<div align="center">

# Welcome to Griffon

</div>

This is the official repo of the Griffon series (v1 & v2). Griffon is the first high-resolution (over 1K) LVLM capable of localizing everything you are interested in describing the region you specify. In the latest version, Griffon supports visual-language co-referring. You can input an image or some descriptions. Griffon achieves excellent performance in REC, object detection, object counting, visual/phrase grounding, and REG.

---

Griffon: Spelling out All Object Locations at Any Granuality with Large Language Model

[`📕Paper`](https://arxiv.org/abs/2311.14552) [`🌀Usage`](./README_v1.md) [`🤗Model`](https://huggingface.co/JefferyZhan/Griffon/tree/main)


Griffon v2: Advancing Multimodal Perception with High-Resolution Scaling and Visual-Language Co-Referring

[`📕Paper`](https://arxiv.org/abs/2403.09333) 

Griffon-G: Bridging Vision-Language and Vision-Centric Tasks via Large Multimodal Models

[`📕Paper`](https://arxiv.org/abs/2410.16163) [`🤗Model`](https://huggingface.co/collections/JefferyZhan/griffon-g-6729d8d65cd58b3f40e87794)

**Release Griffon-G in the next two weeks!** 

## News
- [x] **`2024.07.01`** 🔥**Griffon has been accepted to ECCV 2024.**
- [x] **`2024.03.15`** 🔥Griffon v2's paper has been released in [`📕Arxiv`](https://arxiv.org/abs/2403.09333).
- [x] **`2024.03.11`** 🔥We are excited to announce the arrival of Griffon v2. Griffion v2 brings fine-grained perception performance to new heights with high-resolution expert-level detection and counting and supports visual-language co-referring. Take a look at our demo first. Paper, codes, demos, and models will be released soon.
- [x] **`2023.12.13`** 🔥Ready to release the Language-prompted Localization Dataset after final approval in [`🤗HuggingFace`](https://huggingface.co/datasets/JefferyZhan/Language-prompted-Localization-Dataset).
- [x] **`2023.12.06`** 🔥Release the inference code and model in [`🤗HuggingFace`](https://huggingface.co/JefferyZhan/Griffon/tree/main).
- [x] **`2023.11.29`** 🔥Paper has been released in [`📕Arxiv`](https://arxiv.org/abs/2311.14552).

## What can Griffon do now?
Griffon v2 can perform localization with free-form text inputs and visual target inputs with locally cropped images now, supporting the tasks shown below. **More quantitative evaluation results can be found in our paper.**
![](./demov2.jpg)

## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main) provides the base codes and pre-trained models.
- [Shikra](https://github.com/shikras/shikra) provides insight of how to organize datasets and some base processed annotations.
- [Llama](https://github.com/facebookresearch/llama) provides the large language model.
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

The data and checkpoint is licensed for research use only. All of them are also restricted to uses that follow the license agreement of LLaVA, LLaMA and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.
