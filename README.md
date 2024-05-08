![](./logo.jpg)

<div align="center">

# Welcome to Griffon

</div>

This is the offical repo of Griffon series (v1 & v2). Griffon is the first high-resolution (over 1K) LVLM capable of localizing everything you are interested in describing the region you specify. In the latest version, Griffon support visual-language co-referring. You can input an image or some descriptions. Griffon achieves excellent performance in REC, object detection, object counting, visual/phrase grounding and REG.

---

Griffon: Spelling out All Object Locations at Any Granuality with Large Language Model

[`📕Paper`](https://arxiv.org/abs/2311.14552) [`🌀Usage`](./README_v1.md) [`🤗Model`](https://huggingface.co/JefferyZhan/Griffon/tree/main)


Griffon v2: Advancing Multimodal Perception with High-Resolution Scaling and Visual-Language Co-Referring

[`📕Paper`](https://arxiv.org/abs/2403.09333) 

## News
- [x] **`2024.03.15`** 🔥Griffon v2's paper has been released in [`📕Arxiv`](https://arxiv.org/abs/2403.09333).
- [x] **`2024.03.11`** 🔥We are excited to announce the arrival of Griffon v2. Griffion v2 brings fine-grained perception performance to new heights with high-resolution expert-level detection and counting, and supports visual-language co-referring. Take a look at our demo first. Paper, codes, demos and models will be released soon.
- [x] **`2023.12.13`** 🔥Ready to release the Language-prompted Localization Dataset after final approval in [`🤗HuggingFace`](https://huggingface.co/datasets/JefferyZhan/Language-prompted-Localization-Dataset).
- [x] **`2023.12.06`** 🔥Release the inference code and model in [`🤗HuggingFace`](https://huggingface.co/JefferyZhan/Griffon/tree/main).
- [x] **`2023.11.29`** 🔥Paper has been released in [`📕Arxiv`](https://arxiv.org/abs/2311.14552).

## What can Griffon do now?
Griffon v2 can perform localization with free-form text inputs and visual target inputs with locally cropped images now, supporting the tasks shown as below.
![](./demov2.jpg)

## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main) provides the base codes and pre-trained models.
- [Shikra](https://github.com/shikras/shikra) provides the insight of how to organize datasets and some base processed annotations.
- [Llama](https://github.com/facebookresearch/llama) provides the large language model.
- [volgachen](https://github.com/volgachen/Awesome-AI-Environment) provides the basic environment setting config.

## Citation
If you find Griffon useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{zhan2023griffon,
      title={Griffon: Spelling out All Object Locations at Any Granularity with Large Language Models}, 
      author={Yufei Zhan and Yousong Zhu and Zhiyang Chen and Fan Yang and Ming Tang and Jinqiao Wang},
      year={2023},
      eprint={2311.14552},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{zhan2024griffon,
      title={Griffon v2: Advancing Multimodal Perception with High-Resolution Scaling and Visual-Language Co-Referring}, 
      author={Yufei Zhan and Yousong Zhu and Hongyin Zhao and Fan Yang and Ming Tang and Jinqiao Wang},
      year={2024},
      eprint={2403.09333},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

The data and checkpoint is licensed for research use only. All of them are also restricted to uses that follow the license agreement of LLaVA, LLaMA and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.
