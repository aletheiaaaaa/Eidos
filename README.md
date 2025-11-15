# Eidos

A masked diffusion transformer for text-to-image generation built with PyTorch.

## Overview

Eidos implements a MaskDiT architecture that generates images from text prompts using flow matching and consistency model training. Images are encoded into latent space using a pre-trained VAE, and text is embedded using CLIP.

## Architecture

- **Masked encoder-decoder transformer** for masked diffusion
- **AdaLN modulation** for text conditioning
- **Patch-based image tokenization** (256x256 images, 2x2 patches)
- **Latent denoising** for compute efficient image generation using SDXL

## Methodology

Diffusion models, which are capable of generating novel images from text prompts, work by gradually denoising a random noise image over many steps, eventually producing a high-quality image. In particular, diffusion transformers (DiTs) leverage a transformer architecture to model the denoising process, which allows them to capture long-range dependencies in the image data, and beat convolutional architectures on various benchmarks. However, diffusion transformers are typically highly compute-intensive, requiring many steps to both train and sample from. To address this, Eidos uses a combination of masked representation learning and consistency modelling to significantly reduce the number of steps required for both training and sampling -- both known techniques in the DiT literature.

### MaskDiT Architecture

Since Eidos is fundamentally an image model, we can exploit the fact that images have strong spatial correlations. Specifically, nearby pixels are often similar in color, which leads to a high degree of redundancy in the image representation. We can take advantage of this by using a masked transformer architecture (similar to a masked autoencoder), where a random subset of image patches are masked out during training, and the model is trained to reconstruct the missing patches from the visible ones in addition to denoising the already visible patches. This is done using an asymmetric encoder-decoder architecture, where a large encoder processes only the visible patches, and a lightweight decoder reconstructs the missing patches. 

Concretely, consider an image divided into N patches, of which a random subset M are masked out. The encoder processes the remaining N-M visible patches, producing a latent representation that captures the global context of the image. The decoder then takes these N-M encoded patches, alongside M mask tokens, and simultaneously performs two objectives:
- it denoises the visible patches, and
- it reconstructs the masked patches.
In particular, the reconstructed masked patches are not denoised, and instead matched to the original noisy latent. This forces the model to learn to infer the missing information from the visible patches, leading to a more robust and efficient representation. After training, we perform an additional finetuning step without masking to further improve the denoising performance.

### Flow-Anchored Consistency Modelling

To speed up sampling, we use a consistency training objective, which is designed to directly learn a mapping from a noisy image to a clean image in a single step. This is done by training the model to produce consistent outputs across different noise levels, so that a partially noised image and a fully noised image both map to the same clean image. Moreover, this is augmented using flow matching, which allows the model to learn the denoising process more robustly by forcing it to learn the underlying denoising process. 

Specifically, since consistency models are trained to denoise images in one step, they effectively learn to "shortcut" the multi-step denoising process of traditional diffusion models. However, this means that the underlying denoising process is only implicitly learnt, which can lead to suboptimal results. By incorporating flow matching, we explicitly model the denoising trajectory, allowing the model to learn a more accurate and robust denoising process.

### Further Reading

- [Fast Training of Diffusion Models with Masked Transformers](https://arxiv.org/pdf/2306.09305v2)
- [Flow-Anchored Consistency Models](https://arxiv.org/pdf/2507.03738)

## Quick Start

Install the dependencies, then edit `main.py` to set paths and hyperparameters. Next run

```python
accelerate launch main.py --config_file eidos/config.yaml
```
which will train the model and save checkpoints to the specified output directory.

## Dependencies

```
torch, transformers, diffusers, webdataset,
img2dataset, h5py, accelerate, einops
```