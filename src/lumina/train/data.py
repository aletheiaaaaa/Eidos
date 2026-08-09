import os
import shutil

import h5py
import torch
from datasets import load_dataset
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from ..utils.configs import DataConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_data(cfg: DataConfig) -> None:
    if not os.path.exists(cfg.save_dir):
        os.mkdir(cfg.save_dir)

    dataset = load_dataset(
        path="Fhrozen/relaion-art", split="train", streaming=True
    ).batch(batch_size=cfg.batch_size)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", use_fast=True
    )
    transform = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ]
    )
    shard_ctr = 0

    all_latents = []
    all_embeddings = []
    latent_ctr = 0

    for batch in tqdm(dataset):
        images = batch["image"]
        captions = batch["dense_caption"]

        for img, text in zip(images, captions):
            img = transform(img.convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                latents = (
                    vae.encode(img * 2 - 1).latent_dist.sample()
                    * vae.config.scaling_factor
                )
                inputs = processor(
                    text=text, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                embeds = clip.get_text_features(**inputs)

            all_latents.append(latents.cpu())
            all_embeddings.append(embeds.cpu())
            latent_ctr += latents.size(0)

        if latent_ctr >= cfg.samples_per_shard:
            shard_file = os.path.join(cfg.save_dir, f"shard_{shard_ctr:05d}.h5")
            with h5py.File(shard_file, "w") as h5f:
                h5f.create_dataset(
                    "latents", data=torch.cat(all_latents, dim=0).numpy()
                )
                h5f.create_dataset(
                    "embeddings", data=torch.cat(all_embeddings, dim=0).numpy()
                )

            shard_ctr += 1
            all_latents = []
            all_embeddings = []
            latent_ctr = 0

    if latent_ctr > 0:
        shard_file = os.path.join(cfg.save_dir, f"shard_{shard_ctr:05d}.h5")
        with h5py.File(shard_file, "w") as h5f:
            h5f.create_dataset("latents", data=torch.cat(all_latents, dim=0).numpy())
            h5f.create_dataset(
                "embeddings", data=torch.cat(all_embeddings, dim=0).numpy()
            )

    for file in os.listdir(cfg.save_dir):
        if not file.endswith(".h5"):
            file_path = os.path.join(cfg.save_dir, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)


class H5Dataset(Dataset):
    def __init__(self, data_dir: str) -> None:
        self.data_files = sorted(
            os.path.join(data_dir, file)
            for file in os.listdir(data_dir)
            if file.endswith(".h5")
        )
        self.shard_lengths = []
        for file in self.data_files:
            with h5py.File(file, "r") as h5f:
                self.shard_lengths.append(h5f["latents"].shape[0])
        self.cum_len = torch.cat(
            [torch.tensor([0]), torch.cumsum(torch.tensor(self.shard_lengths), dim=0)]
        )
        self.num_shards = len(self.data_files)

        self.current = -1
        self.latents = None
        self.embeddings = None

    def load_shard(self, shard_idx: int) -> None:
        with h5py.File(self.data_files[shard_idx], "r") as h5f:
            self.latents = h5f["latents"][:]
            self.embeddings = h5f["embeddings"][:]

        shuf_idx = torch.randperm(self.latents.shape[0])
        self.latents = self.latents[shuf_idx.numpy()]
        self.embeddings = self.embeddings[shuf_idx.numpy()]

    def shard_perm(self) -> None:
        perm = torch.randperm(self.num_shards)
        self.shard_lengths = [self.shard_lengths[i] for i in perm]
        self.data_files = [self.data_files[i] for i in perm]
        self.cum_len = torch.cat(
            [torch.tensor([0]), torch.cumsum(torch.tensor(self.shard_lengths), dim=0)]
        )
        self.current = -1

    def __len__(self) -> int:
        return sum(self.shard_lengths)

    def __getitem__(self, idx: int):
        shard_idx = torch.searchsorted(self.cum_len, idx, right=True).item() - 1

        if self.latents is None or shard_idx != self.current:
            self.load_shard(shard_idx)
            self.current = shard_idx

        idx = idx - self.cum_len[shard_idx].item()

        latent = torch.from_numpy(self.latents[idx])
        embedding = torch.from_numpy(self.embeddings[idx])
        return latent, embedding
