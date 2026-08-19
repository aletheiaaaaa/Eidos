# Lumina

Lumina is a few-step latent diffusion transformer (DiT) trained using the MeanFlow objective, that I built to learn how DiTs work. Uses SD-VAE for image latents and CLIP for text embeddings.

## Usage

Edit `config.yaml`, then 

```sh
uv sync
lumina data 
accelerate launch --config_file accelerate.yaml -m lumina.cli train
lumina generate "a lighthouse at dusk" 
```

## Config

Every key in `config.yaml` is optional and falls back to the defaults in `src/lumina/configs.py`; use `lumina config` to print the full configuration as seen by the model.

You can override single values from the file via
```sh
accelerate launch --config_file accelerate.yaml -m lumina.cli --set train.lr=3e-4 --set diffuser.dit.n_layers=12 train
```
To use a different config, make use of the `--config` kewyord argument.

## Guidance

To use classifier-free guidance, use the `-g` or `--guidance` flags

```sh
lumina generate "a lighthouse at dusk" -g 3.0   # -g 1.0 turns guidance off
```
