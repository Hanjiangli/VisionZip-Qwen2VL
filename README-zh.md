# VisionZip-Qwen2VL：使 VisionZip 能够在 Qwen2VL 上运行

本项目旨在将 [VisionZip](https://github.com/dvlab-research/VisionZip) 方法适配到 [Qwen2VL](https://github.com/QwenLM/Qwen2.5-VL) 模型上。VisionZip 能够智能地减少视觉语言模型 (VLM) 处理的视觉token数量，从而在性能影响最小的情况下显著提高性能。 本适配项目能够让 Qwen2VL 用户也能享受到这些优势。

**声明：** 我不是 VisionZip 的原始作者。 本项目是对他们优秀工作的修改和适配，使其能够支持 Qwen2VL 模型。 VisionZip 算法的核心功劳归功于原始作者。 如果您认为本适配项目对您有所帮助，请引用他们的工作。

## 目录
1. [新闻](#新闻)
2. [亮点](#亮点)
3. [安装](#安装)
4. [快速开始](#快速开始)
5. [引用](#引用)
6. [致谢](#致谢)
7. [许可](#许可)
      
## 新闻
- [1] [2025.02.23] 首次发布：VisionZip 与 Qwen2VL 的集成。

## 亮点

- **为 Qwen2VL 提供 VisionZip：** 将 VisionZip 的性能优势（降低计算量、加快推理速度）带给 Qwen2VL 模型。
- **更快的推理速度：** 减少处理的视觉token数量，从而加快 Qwen2VL 的推理速度。
- **最小的性能影响：** 旨在最大限度地减少因视觉token减少而导致的性能下降。
- **易于集成：** 设计为能够相对简单地集成到现有的 Qwen2VL 工作流程中。

## 安装

**安装 Qwen2VL 环境：** 按照官方说明安装 Qwen2VL 环境即可。
    [Qwen2VL](https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#quickstart)。


## 快速开始

以下示例演示如何将 VisionZip 集成到您的 Qwen2VL 流程中：

```
# 由于模型并行性设置,请务必指定 CUDA 型号,
# 否则可能在代码中出现变量 image_mask 和 image_embeds 不在同一 device 上的情况
CUDA_VISIBLE_DEVICES=0 python example.py
```

## 引用

如果您在您的研究中使用 VisionZip-Qwen2VL，请引用原始的 VisionZip 论文：

```
@article{yang2024visionzip,
  title={VisionZip: Longer is Better but Not Necessary in Vision Language Models},
  author={Yang, Senqiao and Chen, Yukang and Tian, Zhuotao and Wang, Chengyao and Li, Jingyao and Yu, Bei and Jia, Jiaya},
  journal={arXiv preprint arXiv:2412.04467},
  year={2024}
}
```

## 致谢

- 我要衷心感谢 [VisionZip](https://github.com/dvlab-research/VisionZip) 的作者，感谢他们的出色工作。 本项目建立在他们的研究之上，并使其可供 Qwen2VL 用户使用。

- 我还要感谢 [Qwen2VL](https://qwen.modelscope.cn/) 的开发人员，感谢他们创建了一个强大而通用的视觉语言模型。

## 许可

- VisionZip-Qwen2VL 遵循 Apache License 2.0 协议。
```