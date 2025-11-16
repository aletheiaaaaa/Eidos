import os
import csv
import h5py
import shutil
import webdataset as wds
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from diffusers import AutoencoderKL
from img2dataset import download
from tqdm import tqdm

from .configs import DataConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_data(cfg: DataConfig) -> None:
    if not os.path.exists(cfg.save_dir):
        os.mkdir(cfg.save_dir)

    dataset = load_dataset(path=cfg.dataset_path, split="train", streaming=True).batch(batch_size=cfg.batch_size)
    vae = AutoencoderKL.from_pretrained(cfg.vae).to(device).eval()
    clip = CLIPModel.from_pretrained(cfg.clip).to(device).eval()
    processor = CLIPProcessor.from_pretrained(cfg.clip, use_fast=True)
    shard_ctr = 0

    all_latents = []
    all_embeddings = []

    for batch in tqdm(dataset):
        with open(os.path.join(cfg.save_dir, "batch.csv"), "w", newline="") as file:
            writer = csv.DictWriter(file, batch.keys())
            writer.writeheader()

            for i in range(len(batch[list(batch.keys())[0]])):
                row = {key: batch[key][i] for key in batch.keys()}
                writer.writerow(row)

        download(
            url_list=os.path.join(cfg.save_dir, "batch.csv"),
            image_size=cfg.img_size,
            output_folder=cfg.save_dir,
            processes_count=16,
            thread_count=256,
            resize_mode="center_crop",
            output_format="webdataset",
            input_format="csv",
            url_col=cfg.url_col,
            caption_col=cfg.caption_col,
            distributor="multiprocessing"
        )

        files = os.listdir(cfg.save_dir)
        files = [os.path.join(cfg.save_dir, file) for file in files if file.endswith(".tar")]

        images = wds.WebDataset(files).decode("pil").to_tuple("jpg;png", "json").map_tuple(transforms.ToTensor(), lambda x: x["caption"])

        dataloader = DataLoader(images, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        for img, text in dataloader:
            img = img.to(device)
            with torch.no_grad():
                latents = vae.encode(img * 2 - 1).latent_dist.sample() * vae.config.scaling_factor
                inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
                embeds = clip.get_text_features(**inputs)

            all_latents.append(latents.squeeze(0).cpu())
            all_embeddings.append(embeds.squeeze(0).cpu())

        if len(all_latents) >= cfg.samples_per_shard:
            shard_file = os.path.join(cfg.save_dir, f"shard_{shard_ctr:05d}.h5")
            with h5py.File(shard_file, "w") as h5f:
                h5f.create_dataset("latents", data=torch.stack(all_latents).numpy())
                h5f.create_dataset("embeddings", data=torch.stack(all_embeddings).numpy())
            
            shard_ctr += 1
            all_latents = []
            all_embeddings = []

    if len(all_latents) > 0:
        shard_file = os.path.join(cfg.save_dir, f"shard_{shard_ctr:05d}.h5")
        with h5py.File(shard_file, "w") as h5f:
            h5f.create_dataset("latents", data=torch.stack(all_latents).numpy())
            h5f.create_dataset("embeddings", data=torch.stack(all_embeddings).numpy())

    for file in os.listdir(cfg.save_dir):
        if not file.endswith(".h5"):
            file_path = os.path.join(cfg.save_dir, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)

class H5Dataset(Dataset):
    def __init__(self, data_dir: str) -> None:
        self.data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".h5")]
        self.shard_lengths = [h5py.File(file, "r")["latents"].shape[0] for file in self.data_files]
        self.cum_len = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(self.shard_lengths), dim=0)])
        self.num_shards = len(self.data_files)

        self.current = -1
        self.latents = None
        self.embeddings = None
    
    def load_shard(self, shard_idx: int) -> None:
        with h5py.File(self.data_files[shard_idx], "r") as h5f:
            self.latents = h5f["latents"][:]
            self.embeddings = h5f["embeddings"][:]

        shuf_idx = torch.randperm(self.latents.shape[0])
        self.latents = self.latents[shuf_idx]
        self.embeddings = self.embeddings[shuf_idx]

    def shard_perm(self) -> None:
        perm = torch.randperm(self.num_shards)
        self.shard_lengths = [self.shard_lengths[i] for i in perm]
        self.data_files = [self.data_files[i] for i in perm]
        self.cum_len = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(self.shard_lengths), dim=0)])
        self.current = -1

    def __len__(self) -> int:
        return sum(self.shard_lengths)

    def __getitem__(self, idx: int):
        shard_idx = torch.searchsorted(self.cum_len, idx, right=True).item()
        
        if self.latents is None or shard_idx != self.current:
            self.load_shard(shard_idx)
            self.current = shard_idx
        
        idx = idx - self.cum_len[shard_idx - 1].item()

        latent = torch.from_numpy(self.latents[idx])
        embedding = torch.from_numpy(self.embeddings[idx])
        return latent, embedding