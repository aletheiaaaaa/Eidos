import os

import h5py
import torch
from datasets import load_dataset
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .configs import DataConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _write(save_dir: str, shard_idx: int, latents: list, embeddings: list) -> None:
    shard_file = os.path.join(save_dir, f"shard_{shard_idx:05d}.h5")
    with h5py.File(shard_file, "w") as h5f:
        h5f.create_dataset("latents", data=torch.cat(latents, dim=0).numpy())
        h5f.create_dataset("embeddings", data=torch.cat(embeddings, dim=0).numpy())


def _encode(images, captions, vae, clip, processor, transform):
    pixels, texts = [], []
    for img, text in zip(images, captions):
        try:
            pixels.append(transform(img.convert("RGB")))
        except (OSError, ValueError):
            continue
        texts.append(text)

    if not pixels:
        return None, None

    pixels = torch.stack(pixels).to(device)

    with torch.no_grad():
        latents = (
            vae.encode(pixels * 2 - 1).latent_dist.sample() * vae.config.scaling_factor
        )
        inputs = processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        embeds = clip.get_text_features(**inputs)

    return latents.half().cpu(), embeds.half().cpu()


def process_data(cfg: DataConfig) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)

    dataset = load_dataset(path=cfg.dataset, split=cfg.split, streaming=True)
    if cfg.max_samples > 0:
        dataset = dataset.take(cfg.max_samples)
    dataset = dataset.batch(batch_size=cfg.stream_batch_size)
    vae = AutoencoderKL.from_pretrained(cfg.vae).to(device).eval()
    clip = CLIPModel.from_pretrained(cfg.clip).to(device).eval()
    processor = CLIPProcessor.from_pretrained(cfg.clip, use_fast=True)
    transform = transforms.Compose(
        [
            transforms.Resize(cfg.resolution),
            transforms.CenterCrop(cfg.resolution),
            transforms.ToTensor(),
        ]
    )

    shard_ctr = 0
    all_latents = []
    all_embeddings = []
    latent_ctr = 0

    for batch in tqdm(dataset, desc="stream"):
        images = batch["image"]
        captions = batch["dense_caption"]

        for start in tqdm(
            range(0, len(images), cfg.encode_batch_size), desc="encode", leave=False
        ):
            stop = start + cfg.encode_batch_size
            latents, embeds = _encode(
                images[start:stop],
                captions[start:stop],
                vae,
                clip,
                processor,
                transform,
            )
            if latents is None:
                continue

            all_latents.append(latents)
            all_embeddings.append(embeds)
            latent_ctr += latents.size(0)

            if latent_ctr >= cfg.samples_per_shard:
                _write(cfg.save_dir, shard_ctr, all_latents, all_embeddings)

                shard_ctr += 1
                all_latents = []
                all_embeddings = []
                latent_ctr = 0

    if latent_ctr > 0:
        _write(cfg.save_dir, shard_ctr, all_latents, all_embeddings)


class H5Dataset(Dataset):
    def __init__(self, data_dir: str, seed: int = 0) -> None:
        self.base_files = sorted(
            os.path.join(data_dir, file)
            for file in os.listdir(data_dir)
            if file.endswith(".h5")
        )
        if not self.base_files:
            raise FileNotFoundError(f"no .h5 shards found in {data_dir}")

        self.base_lengths = []
        for file in self.base_files:
            with h5py.File(file, "r") as h5f:
                self.base_lengths.append(h5f["latents"].shape[0])

        self.num_shards = len(self.base_files)
        self.seed = seed
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

        gen = torch.Generator().manual_seed(self.seed + epoch)
        order = torch.randperm(self.num_shards, generator=gen).tolist()

        self.data_files = [self.base_files[i] for i in order]
        self.shard_lengths = [self.base_lengths[i] for i in order]
        self.cum_len = torch.cat(
            [torch.tensor([0]), torch.cumsum(torch.tensor(self.shard_lengths), dim=0)]
        )

        self.current = -1
        self.latents = None
        self.embeddings = None
        self.perm = None

    def load_shard(self, shard_idx: int) -> None:
        with h5py.File(self.data_files[shard_idx], "r") as h5f:
            self.latents = h5f["latents"][:]
            self.embeddings = h5f["embeddings"][:]

        gen = torch.Generator().manual_seed(
            self.seed + self.epoch * 1_000_003 + shard_idx
        )
        self.perm = torch.randperm(self.latents.shape[0], generator=gen).numpy()

    def __len__(self) -> int:
        return sum(self.shard_lengths)

    def __getitem__(self, index: int):
        shard_idx = int(torch.searchsorted(self.cum_len, index, right=True).item()) - 1

        if self.latents is None or shard_idx != self.current:
            self.load_shard(shard_idx)
            self.current = shard_idx

        index = self.perm[index - int(self.cum_len[shard_idx].item())]

        latent = torch.from_numpy(self.latents[index]).float()
        embedding = torch.from_numpy(self.embeddings[index]).float()
        return latent, embedding
