# Eidos

A masked diffusion transformer for text-to-image generation built with PyTorch.

## Overview

Eidos implements a MaskDiT architecture that generates images from text prompts using flow matching and consistency model training. Images are encoded into latent space using a pre-trained VAE, and text is embedded using CLIP.

## Architecture

- **Masked encoder-decoder transformer** for masked diffusion
- **AdaLN modulation** for text conditioning
- **Patch-based image tokenization** (256x256 images, 2x2 patches)
- **Latent denoising** for compute efficient image generation using SDXL

## Training Methodology

### MaskDiT

TODO

### Flow-Anchored Consistency Modeling

TODO

## Quick Start

TODO

## Dependencies

```
torch, transformers, diffusers, webdataset,
img2dataset, h5py, accelerate, einops
```