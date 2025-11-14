import torch
import h5py
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from model import Diffuser
from configs import TrainConfig
from data import H5Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model: Diffuser, cfg: TrainConfig) -> None:
    dataset = H5Dataset(cfg.data_path)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, drop_last=True)

    optimizer = optim.AdamW(model.maskdit.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    accelerator = Accelerator(mixed_precision="bf16" if torch.cuda.is_available() else "no")
    model.maskdit, optimizer = accelerator.prepare(model.maskdit, optimizer)

    for epoch in range(cfg.epochs):
        dataset.shard_perm()
        for latent, embedding in tqdm(dataloader):
            latent = latent.to(device)
            embedding = embedding.to(device)

            noise = torch.randn_like(latent)

            sigma = (cfg.p_std * torch.randn(latent.size(0), device=device) + cfg.p_mean).exp()
            sigma = torch.arctan(sigma) * (2 / torch.pi)
            time = (1 - sigma)

            aux = torch.ones_like(time).to(device)

            noisy_latent = time.reshape(-1, 1, 1, 1) * latent + (1 - time.reshape(-1, 1, 1, 1)) * noise

            fm_target = latent - noise
            fm_pred = model.maskdit(noisy_latent, embedding, time.flatten(), time.flatten())

            fm_mse = (fm_pred - fm_target).pow(2).mean(dim=(list(range(1, fm_pred.dim()))))
            fm_cos = (1 - F.cosine_similarity(fm_pred.flatten(1), fm_target.flatten(1), dim=1))
            fm_loss = fm_mse + fm_cos

            (cm_pred, cm_grad) = torch.func.jvp(
                lambda x, t: model.maskdit(x, embedding, t, aux.flatten()),
                (noisy_latent, time.flatten()),
                (fm_target, aux.flatten())
            )

            cm_sg = cm_pred.detach()
            cm_grad = cm_grad.detach()

            res = cm_sg - (fm_target + (1 - time.reshape(-1, 1, 1, 1)) * cm_grad)
            alpha = 1 - time.reshape(-1, 1, 1, 1).pow(0.5)
            cm_target = cm_grad - alpha * res.clamp(-1, 1)

            beta = torch.cos(time * torch.pi / 2)
            cm_loss = (cm_pred - cm_target).pow(2).mean(dim=(list(range(1, cm_pred.dim()))))
            cm_loss = beta * cm_loss / (cm_loss + 1e-3).pow(0.5).detach()

            loss = fm_loss.mean() + cm_loss.mean()

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

        if epoch % cfg.save_interval == 0 or epoch == cfg.epochs - 1:
            torch.save(model.maskdit.state_dict(), f"{cfg.output_dir}/eidos_epoch_{epoch:04d}.pt")
