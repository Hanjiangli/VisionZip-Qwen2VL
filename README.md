# VisionZip-Qwen2VL: Enabling VisionZip for Qwen2VL

<div align="center">
[简体中文](https://github.com/Hanjiangli/VisionZip-Qwen2VL/README-zh.md)
</div>

This project aims to adapt the [VisionZip](https://github.com/dvlab-research/VisionZip) method to the [Qwen2VL](https://github.com/QwenLM/Qwen2.5-VL) model. VisionZip intelligently reduces the number of visual tokens processed by a Vision Language Model (VLM), leading to significant performance improvements with minimal performance impact.  This adaptation project enables Qwen2VL users to enjoy these benefits.

**Disclaimer:** I am not the original author of VisionZip. This project is a modification and adaptation of their excellent work to support the Qwen2VL model. All credit for the core VisionZip algorithm goes to the original authors.  Please cite their work if you find this adaptation useful.

## TABLE OF CONTENTS
1. [News](#news)
2. [Highlights](#highlights)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Citation](#citation)
6. [Acknowledgement](#acknowledgement)
7. [License](#license)

## News
- [1] [2025.02.23] Initial release: Integration of VisionZip with Qwen2VL.

## Highlights

- **Provides VisionZip for Qwen2VL:** Brings the performance benefits of VisionZip (reduced computation, faster inference) to the Qwen2VL model.
- **Faster Inference Speed:** Reduces the number of visual tokens processed, thereby accelerating the inference speed of Qwen2VL.
- **Minimal Performance Impact:** Aims to minimize performance degradation due to the reduction of visual tokens.
- **Easy Integration:** Designed to be relatively easily integrated into existing Qwen2VL workflows.

## Installation

**Install Qwen2VL Environment:**  Follow the official instructions to install the Qwen2VL environment.
    [Qwen2VL](https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#quickstart).

## Quick Start

The following example demonstrates how to integrate VisionZip into your Qwen2VL process:

```
# Due to model parallelism settings, it is essential to specify the CUDA device.
# Otherwise, the variables image_mask and image_embeds may end up on different devices in the code.
CUDA_VISIBLE_DEVICES=0 python example.py
```

## Citation

If you use VisionZip-Qwen2VL in your research, please cite the original VisionZip paper:

```
@article{yang2024visionzip,
  title={VisionZip: Longer is Better but Not Necessary in Vision Language Models},
  author={Yang, Senqiao and Chen, Yukang and Tian, Zhuotao and Wang, Chengyao and Li, Jingyao and Yu, Bei and Jia, Jiaya},
  journal={arXiv preprint arXiv:2412.04467},
  year={2024}
}
```

## Acknowledgement

- I would like to express my sincere gratitude to the authors of [VisionZip](https://github.com/dvlab-research/VisionZip) for their excellent work. This project builds upon their research and makes it accessible for Qwen2VL users.

- I also thank the developers of [Qwen2VL](https://qwen.modelscope.cn/) for creating a powerful and versatile Vision Language Model.

## License

- VisionZip-Qwen2VL is licensed under the Apache License 2.0.
```
