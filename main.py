from lumina.model import Diffuser
from lumina.configs import DiffuserConfig, TrainConfig, DataConfig, DiTConfig
from lumina.data import process_data
from lumina.train import finetune, train

data_cfg = DataConfig(
    img_size=256,
    dataset_path="laion/relaion-art",
    batch_size=16384,
    save_dir="./data",
    vae="stabilityai/sd-vae-ft-mse",
    clip="openai/clip-vit-large-patch14",
    samples_per_shard=100000,
    url_col="URL",
    caption_col="TEXT",
)

train_cfg = TrainConfig(
    p_mean=-0.8,
    p_std=1.6,
    train_epochs=200,
    finetune_epochs=20,
    batch_size=128,
    train_lr=1e-4,
    finetune_lr=1e-5,
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
    vae="stabilityai/sd-vae-ft-mse",
    clip="openai/clip-vit-large-patch14",
    model_path="",
    dit=DiTConfig(
        d_model=512,
        n_heads=8,
        d_head=64,
        d_mlp=2048,
        n_layers=6
    ),
)

if __name__ == "__main__":
    process_data(data_cfg)
    model = Diffuser(diffuser_cfg)
    train(model, train_cfg)
    finetune(model, train_cfg)