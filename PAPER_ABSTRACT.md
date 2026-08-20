# GeoFSR-GAN: Dual-Domain Spatial-Frequency Adversarial Network for Satellite Imagery Super-Resolution and Downstream Task Guidance

**Abstract** — High-resolution (HR) satellite imagery is fundamental to remote sensing applications such as urban building extraction, road network mapping, and disaster damage assessment. However, hardware transmission constraints and atmospheric optical attenuation often degrade satellite sensors to low-resolution (LR) observations. Conventional spatial-domain super-resolution (SR) networks struggle to synthesize fine-grained structural textures, frequently producing oversmoothed high-frequency details. In this work, we propose **GeoFSR-GAN**, a novel dual-domain adversarial framework combining spatial residual dense encoding with 2D Discrete Wavelet Transform (DWT) frequency decomposition.

Key architectural innovations of GeoFSR-GAN include:
1. **Dual Spatial-Frequency Feature Decomposition**: Concurrent spatial feature extraction via Residual-in-Residual Dense Blocks (RRDB) and spectral sub-band extraction ($LL, LH, HL, HH$) via 2D Haar DWT.
2. **Lightweight Cross-Attention Fusion**: An adaptive cross-attention mechanism that bidirectionally aligns spatial feature maps with frequency representations under $O(N \cdot N/r^2)$ computational complexity suitable for edge and CPU/GPU deployments.
3. **Multi-Band Wavelet Frequency Loss**: A sub-band weighted loss function ($\mathcal{L}_{\text{freq}}$) emphasizing high-frequency spectral bands ($LH, HL, HH$) to enforce high-fidelity texture synthesis.
4. **Dual-Domain PatchGAN Discriminators**: Independent spatial and DWT spectral PatchGAN discriminators that jointly constrain spatial realism and high-frequency power distribution.
5. **Downstream Task Guidance**: A lightweight 4-level UNet segmentation guidance module constraining the generator to preserve critical land-cover boundaries for downstream GIS tasks.

Empirical evaluations demonstrate that GeoFSR-GAN outperforms traditional bicubic interpolation and standard spatial CNN baselines in both perceptual structural similarity (SSIM) and downstream building footprint extraction (mIoU), while demonstrating superior robustness under atmospheric noise, spatial blur, and sensor compression degradations.
