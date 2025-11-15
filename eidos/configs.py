from dataclasses import dataclass

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
    d_caption: int
    n_channels: int
    vae: str
    clip: str
    model_path: str

    encoder: DiTConfig
    decoder: DiTConfig

@dataclass
class DataConfig:
    img_size: int
    dataset_path: str
    batch_size: int
    save_dir: str
    vae: str
    clip: str
    samples_per_shard: int

@dataclass
class TrainConfig:
    p_mean: float
    p_std: float
    train_epochs: int
    finetune_epochs: int
    batch_size: int
    train_lr: float
    finetune_lr: float
    weight_decay: float
    mask_freq: float
    mae_weight: float
    data_path: str
    save_interval: int
    output_dir: str


