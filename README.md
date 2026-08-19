# lumina

A latent diffusion transformer (DiT) trained with mean-flow matching, so a
sample takes one or two steps instead of a full denoising schedule.

Images are encoded once with a frozen SD VAE and captions with CLIP ViT-L/14;
training then runs entirely in latent space.

## Install

```sh
uv sync
```

## Use

Everything reads `config.yaml` at the project root.

```sh
lumina data                            # encode the dataset into latent shards
lumina train                           # train the denoiser
lumina generate "a lighthouse at dusk" # sample images into ./samples
```

Multi-GPU:

```sh
accelerate launch --config_file accelerate.yaml -m lumina.cli train
```

## Config

`config.yaml` has three sections — `data`, `train`, and `diffuser` (with a
nested `dit`). Every key is optional and falls back to the defaults in
`src/lumina/configs.py`. Use `lumina config` to print what actually resolved.

Override single values without editing the file:

```sh
lumina --set train.lr=3e-4 --set diffuser.dit.n_layers=12 train
```

Point `-c/--config` elsewhere to keep several configs around.

Note that `diffuser.img_size` is the *latent* resolution: `data.resolution // 8`
for the SD VAE.

## Training

Checkpoints land in `train.output_dir` every `train.save_interval` epochs and
carry the model, optimizer, scheduler, EMA and epoch, so a run picks up exactly
where it stopped:

```sh
lumina train --resume checkpoints/checkpoint_000050.pt
```

At the end of a run `model.pt` is written with the EMA weights (or the raw
weights if `train.ema_decay` is 0). That is the file `generate` wants.

Set `train.wandb_project` to log loss, learning rate and gradient norm to
Weights & Biases. With `train.sample_interval` and `train.sample_prompts` set,
images are sampled from the EMA weights every N epochs into
`output_dir/samples/` and attached to the run.

## Guidance

`train.p_uncond` is the rate at which captions are replaced by a learned null
embedding during training, which is what makes classifier-free guidance work at
sampling time:

```sh
lumina generate "a lighthouse at dusk" -g 3.0   # -g 1.0 turns guidance off
```

## Layout

```
src/lumina/
  cli.py           entry point
  configs.py       config dataclasses + YAML loading
  data.py          dataset encoding, shard-backed H5Dataset
  train.py         training loop, EMA
  nn/model.py      DiT denoiser, Diffuser sampling wrapper
  nn/components.py attention, MLP, adaLN-zero modulation, embeddings
```
