import math
import os

import torch
from accelerate import Accelerator
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .configs import DataConfig, TrainConfig
from .data import H5Dataset
from .nn.model import DiT


def train(model: DiT, cfg: TrainConfig, data: DataConfig):
    accel = Accelerator(
        mixed_precision=cfg.mixed_precision if torch.cuda.is_available() else "no"
    )
    device = accel.device

    dataset = H5Dataset(data_dir=data.save_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    def lr_lambda(epoch: int) -> float:
        if epoch < cfg.n_warmup:
            return (epoch + 1) / max(cfg.n_warmup, 1)
        progress = (epoch - cfg.n_warmup) / max(cfg.n_epochs - cfg.n_warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    model, dataloader, optimizer, scheduler = accel.prepare(
        model, dataloader, optimizer, scheduler
    )

    os.makedirs(cfg.output_dir, exist_ok=True)

    def sample_r_t(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = (torch.randn(batch, 2, device=device) * cfg.p_std + cfg.p_mean).sigmoid()
        r, t = s.amin(dim=-1), s.amax(dim=-1)

        return torch.where(torch.rand(batch, device=device) >= cfg.p_ratio, t, r), t

    def step(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        def adaptive_l2(error: torch.Tensor) -> torch.Tensor:
            d = error.float().pow(2).flatten(1).mean(-1)
            w = (d.detach() + 1e-3).pow(-0.5)

            return (w * d).mean()

        b = x.shape[0]
        r, t = sample_r_t(b)
        rb = r.view(-1, *([1] * (x.dim() - 1)))
        tb = t.view(-1, *([1] * (x.dim() - 1)))

        eps = torch.randn_like(x)
        z = (1.0 - tb) * x + tb * eps
        v = eps - x

        u, dudt = torch.func.jvp(  # ty: ignore
            lambda z, r, t: model(z, c, r, t),
            (z, r, t),
            (v, torch.zeros_like(r), torch.ones_like(t)),
        )

        tgt = (v - (tb - rb) * dudt).detach()

        return adaptive_l2(u - tgt)

    def save(name: str) -> None:
        accel.wait_for_everyone()
        if accel.is_main_process:
            accel.save(
                accel.unwrap_model(model).state_dict(),
                os.path.join(cfg.output_dir, name),
            )

    for epoch in tqdm(range(cfg.n_epochs), desc="epoch"):
        model.train()
        total, seen = 0.0, 0

        for latent, emb in tqdm(dataloader, desc=f"epoch {epoch}", leave=False):
            optimizer.zero_grad()
            loss = step(latent, emb)
            accel.backward(loss)
            optimizer.step()

            total += loss.item()
            seen += 1

        scheduler.step()
        dataset.shard_perm()

        accel.print(f"epoch {epoch}: loss {total / max(seen, 1):.4f}")

        if cfg.save_interval > 0 and (epoch + 1) % cfg.save_interval == 0:
            save(f"model_{epoch + 1:06d}.pt")

    save("model.pt")
