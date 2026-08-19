import json
import os
import time

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


def _load_state(save_dir: str) -> tuple[int, int]:
    path = os.path.join(save_dir, "state.json")
    if not os.path.exists(path):
        return 0, 0

    with open(path) as f:
        state = json.load(f)

    return state["shards"], state["consumed"]


def _save_state(save_dir: str, shards: int, consumed: int) -> None:
    path = os.path.join(save_dir, "state.json")
    tmp = f"{path}.tmp"

    with open(tmp, "w") as f:
        json.dump({"shards": shards, "consumed": consumed}, f)

    os.replace(tmp, path)


def _stream(cfg: DataConfig, encoders, shard_ctr: int, consumed: int) -> tuple[int, int]:
    vae, clip, processor, transform = encoders

    dataset = load_dataset(path=cfg.dataset, split=cfg.split, streaming=True)
    if consumed:
        dataset = dataset.skip(consumed)
    if cfg.max_samples > 0:
        remaining = cfg.max_samples - consumed
        if remaining <= 0:
            return shard_ctr, consumed
        dataset = dataset.take(remaining)

    all_latents = []
    all_embeddings = []
    images = []
    captions = []
    pending = 0
    count = 0

    def flush() -> None:
        nonlocal shard_ctr, consumed, pending, count

        _write(cfg.save_dir, shard_ctr, all_latents, all_embeddings)

        shard_ctr += 1
        consumed += pending
        pending = 0
        count = 0

        _save_state(cfg.save_dir, shard_ctr, consumed)
        all_latents.clear()
        all_embeddings.clear()

    def drain() -> None:
        nonlocal count

        if not images:
            return

        latents, embeds = _encode(images, captions, vae, clip, processor, transform)
        images.clear()
        captions.clear()

        if latents is None:
            return

        all_latents.append(latents)
        all_embeddings.append(embeds)
        count += latents.size(0)

        if count >= cfg.samples_per_shard:
            flush()

    progress = tqdm(initial=consumed, total=cfg.max_samples or None, desc="encode")

    try:
        for example in dataset:
            images.append(example["image"])
            captions.append(example["dense_caption"])
            pending += 1
            progress.update(1)

            if len(images) >= cfg.encode_batch_size:
                drain()

        drain()
        if all_latents:
            flush()
    except BaseException:
        pending -= len(images)
        images.clear()
        captions.clear()

        if all_latents:
            flush()
        raise
    finally:
        progress.close()

    return shard_ctr, consumed


def process_data(cfg: DataConfig) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)

    encoders = (
        AutoencoderKL.from_pretrained(cfg.vae).to(device).eval(),
        CLIPModel.from_pretrained(cfg.clip).to(device).eval(),
        CLIPProcessor.from_pretrained(cfg.clip, use_fast=True),
        transforms.Compose(
            [
                transforms.Resize(cfg.resolution),
                transforms.CenterCrop(cfg.resolution),
                transforms.ToTensor(),
            ]
        ),
    )

    shard_ctr, consumed = _load_state(cfg.save_dir)
    if consumed:
        print(f"resuming after {consumed} examples, {shard_ctr} shards")

    for attempt in range(cfg.max_retries + 1):
        try:
            _stream(cfg, encoders, shard_ctr, consumed)
            return
        except Exception as err:
            shard_ctr, consumed = _load_state(cfg.save_dir)
            if attempt == cfg.max_retries:
                raise

            wait = min(300, 15 * 2**attempt)
            print(f"stream failed ({type(err).__name__}: {err})")
            print(f"retry {attempt + 1}/{cfg.max_retries} in {wait}s from {consumed}")
            time.sleep(wait)


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
