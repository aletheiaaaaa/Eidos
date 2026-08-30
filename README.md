# Lumina

Lumina is a few-step latent diffusion transformer (DiT) trained using the MeanFlow objective, that I built to learn how DiTs work. Uses SD-VAE for image latents and CLIP for text embeddings.

## Usage

Edit `config.yaml`, then 

```sh
uv sync
uv run lumina data 
uv run accelerate launch --config_file accelerate.yaml -m lumina.cli train
uv run lumina generate "a lighthouse at dusk" 
```

## Data

Captions come from the `dense_caption` field, which averages ~128 CLIP tokens and so overruns the text encoder's 77-token window on essentially every sample. Only the first sentence is encoded (~33 tokens), which avoids the truncation and keeps the stored embeddings near the short prompts used at generation time — the model conditions on a single pooled vector, so the detail in the rest of the caption is not usable anyway.

## Config

Every key in `config.yaml` is optional and falls back to the defaults in `src/lumina/configs.py`; use `lumina config` to print the full configuration as seen by the model.

To use a different config, make use of the `--config` kewyord argument.

## Guidance

To use classifier-free guidance, use the `-g` or `--guidance` flag

```sh
uv run lumina generate "a lighthouse at dusk" -g 3.0   # -g 1.0 turns guidance off
```
