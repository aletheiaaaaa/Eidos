import torch

from eidos.model import Diffuser
from eidos.configs import DiffuserConfig, TrainConfig, DataConfig
from eidos.data import process_data
from eidos.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_cfg = DataConfig(
    img_size=256,
    dataset_path="laion/relaion-art",
    batch_size=64,
    save_dir="./data",
    vae="stabilityai/sd-vae-ft-mse",
    clip="openai/clip-vit-large-patch14",
    samples_per_shard=1000
)

train_cfg = TrainConfig(
    p_mean=-0.8,
    p_std=1.6,
    epochs=200,
    batch_size=62,
    learning_rate=1e-4,
    weight_decay=1e-2,
    data_path="./data",
    save_interval=50,
    output_dir="./checkpoints"
)

diffuser_cfg = DiffuserConfig(
    img_size=256,
    patch_size=2,
    d_caption=768,
    n_channels=4,
    mask_freq=0.5,
    vae="stabilityai/sd-vae-ft-mse",
    clip="openai/clip-vit-large-patch14",
    model_path="",
    encoder=DiffuserConfig.encoder.__class__(
        d_model=512,
        n_heads=4,
        d_head=64,
        d_mlp=2048,
        n_layers=4
    ),
    decoder=DiffuserConfig.decoder.__class__(
        d_model=256,
        n_heads=4,
        d_head=32,
        d_mlp=1024,
        n_layers=2
    ),
)

if __name__ == "__main__":
    process_data(data_cfg)
    model = Diffuser(diffuser_cfg, device)
    train(model, train_cfg)