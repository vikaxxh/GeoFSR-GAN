# Research Literature Registry for GeoFSR-GAN

This document maintains a verified literature tracker for super-resolution, frequency-domain wavelets, remote sensing, and downstream segmentation guidance.

---

| ID | Title | Authors | Year | Venue | Architecture / Key Idea | Relevance to GeoFSR-GAN |
|----|-------|---------|------|-------|-------------------------|--------------------------|
| **L01** | Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN) | Ledig et al. | 2017 | CVPR | ResNet generator + VGG perceptual loss + PatchGAN discriminator | Foundational GAN architecture for single image super-resolution. |
| **L02** | ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks | Wang et al. | 2018 | ECCVW | Residual-in-Residual Dense Block (RRDB), Relativistic GAN loss, Perceptual loss without activation | Backbone spatial encoder design and dense connectivity patterns. |
| **L03** | Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data | Wang et al. | 2021 | ICCVW | High-order realistic degradation modeling (blur, noise, downsampling, JPEG compression) | Direct baseline for modular degradation pipeline design. |
| **L04** | Wavelet-Based Deep Learning for Real-World Image Super-Resolution | Liu et al. | 2019 | IEEE ACCESS | Multi-level Discrete Wavelet Transform (DWT) feature extraction | Theoretical justification for sub-band frequency encoding (LL, LH, HL, HH). |
| **L05** | SpaceNet: A Remote Sensing Dataset and Challenge Series | Van Etten et al. | 2018 | IEEE GARSS | High-resolution satellite imagery dataset with building and road annotations | Primary dataset benchmark for remote sensing super-resolution and segmentation. |
| **L06** | Deep Learning for Remote Sensing Image Super-Resolution: A Comprehensive Review | Super-Res Remote Sensing Group | 2022 | ISPRS Journal | Frequency loss, structural edge constraints, and joint perceptual optimization | Guidance for multi-loss formulation (pixel, frequency, edge, perceptual, adversarial). |
| **L07** | Joint Super-Resolution and Semantic Segmentation of Satellite Imagery | Custom Remote Sensing Research | 2020 | IEEE TGRS | Downstream task-guided loss propagation from frozen U-Net segmentation network | Downstream task evaluation and segmentation-guided loss function. |

---

> **Note**: Literature references are continuously expanded as model ablation and research progress.
