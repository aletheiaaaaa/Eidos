from dataclasses import dataclass
from diffusers import AutoencoderKL

@dataclass
class DiTConfig:
    d_model: int
    n_heads: int
    d_head: int
    d_mlp: int
    n_layers: int

@dataclass
class DiffuserConfig:
    img_size: int
    patch_size: int
    d_model_wt: int
    n_channels: int
    mask_freq: float
    encoder: DiTConfig
    decoder: DiTConfig
    dropout: float

@dataclass
class DataConfig:
    path: str
    split: str = "train"
    streaming: bool = True
    batch_size: int
    vae: AutoencoderKL

