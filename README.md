# Lumina

Lumina is a few-step latent diffusion transformer (DiT) trained using the MeanFlow objective, that I built to learn how DiTs work. Diffusion runs in DINOv3 representation space rather than a VAE latent space, following the RAE recipe, with CLIP text embeddings conditioning the model through cross-attention.

Images and captions are encoded on the fly while training streams ImageNet-1k, so nothing is written to disk.

## Usage

Edit `config.yaml`, then 

```sh
uv sync
uv run accelerate launch --config_file accelerate.yaml -m lumina.cli train
uv run lumina generate "a photo of a golden retriever"
```

Pixel and latent statistics are estimated from the first `data.n_stat_batches`
batches of the stream at startup, then frozen and saved into every checkpoint, so a
resumed run normalises exactly as the original did.

Both `ILSVRC/imagenet-1k` and the DINOv3 weights are gated on the Hugging Face hub,
so accept their terms and run `hf auth login` first.

## Decoding

Diffusion happens in DINOv3 space, which has no decoder of its own. Training and
latent-space evaluation work without one, but turning latents back into pixels needs
a decoder trained against the frozen encoder; point `diffuser.decoder_path` at it to
enable `generate` and periodic samples during training.

## Config

Every key in `config.yaml` is optional and falls back to `src/lumina/configs.py`; use `lumina config` to print the full configuration used.

To use a different config, make use of the `--config` kewyord argument.

## Guidance

To use classifier-free guidance, use the `-g` or `--guidance` flag

```sh
uv run lumina generate "a photo of a volcano" -g 3.0   # -g 1.0 turns guidance off
```
