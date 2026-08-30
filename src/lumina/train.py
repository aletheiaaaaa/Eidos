import contextlib
import dataclasses
import math
import os

import torch
from accelerate import Accelerator
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .configs import DataConfig, DiffuserConfig, TrainConfig
from .data import H5Dataset
from .nn.model import Diffuser, DiT


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            key: value.detach().clone().float()
            for key, value in model.state_dict().items()
            if value.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            if key in self.shadow:
                self.shadow[key].lerp_(value.detach().float(), 1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow

    def load_state_dict(self, state: dict) -> None:
        for key, value in state.items():
            if key in self.shadow:
                self.shadow[key].copy_(value)

    def weights(self, model: torch.nn.Module) -> dict:
        return {
            key: self.shadow[key].to(value.dtype)
            for key, value in model.state_dict().items()
            if key in self.shadow
        }

    @contextlib.contextmanager
    def averaged(self, model: torch.nn.Module):
        backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if key in self.shadow
        }
        model.load_state_dict(self.weights(model), strict=False)
        try:
            yield model
        finally:
            model.load_state_dict(backup, strict=False)


def train(
    model: DiT,
    cfg: TrainConfig,
    data: DataConfig,
    diffuser: DiffuserConfig | None = None,
    resume: str | None = None,
):
    accel = Accelerator(
        mixed_precision=cfg.mixed_precision if torch.cuda.is_available() else "no",
        log_with="wandb" if cfg.wandb_project else None,
    )
    device = accel.device

    if cfg.wandb_project:
        accel.init_trackers(
            cfg.wandb_project,
            config={"train": dataclasses.asdict(cfg), "data": dataclasses.asdict(data)},
        )

    dataset = H5Dataset(data_dir=data.save_dir, seed=cfg.seed)
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

    net = accel.unwrap_model(model)
    ema = EMA(net, cfg.ema_decay) if cfg.ema_decay > 0 else None

    os.makedirs(cfg.output_dir, exist_ok=True)

    start_epoch = 0
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if "model" in state:
            net.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            if ema is not None and "ema" in state:
                ema.load_state_dict(state["ema"])
            start_epoch = state["epoch"]
        else:
            net.load_state_dict(state)
        accel.print(f"resumed from {resume} at epoch {start_epoch}")

    def step(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        def adaptive_l2(error: torch.Tensor) -> torch.Tensor:
            d = error.float().pow(2).flatten(1).mean(-1)
            w = (d.detach() + 1e-3).pow(-0.5)

            return (w * d).mean()

        def sample_r_t(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
            s = (
                torch.randn(batch, 2, device=device) * cfg.p_std + cfg.p_mean
            ).sigmoid()
            r, t = s.amin(dim=-1), s.amax(dim=-1)

            return torch.where(torch.rand(batch, device=device) >= cfg.p_ratio, t, r), t

        b = x.shape[0]
        r, t = sample_r_t(b)
        rb = r.view(-1, *([1] * (x.dim() - 1)))
        tb = t.view(-1, *([1] * (x.dim() - 1)))

        drop = torch.rand(b, device=device) < cfg.p_uncond

        eps = torch.randn_like(x)
        z = (1.0 - tb) * x + tb * eps
        v = eps - x

        u, dudt = torch.func.jvp(  # ty: ignore
            lambda z, r, t: model(z, c, r, t, drop),
            (z, r, t),
            (v, torch.zeros_like(r), torch.ones_like(t)),
        )

        tgt = (v - (tb - rb) * dudt).detach()

        return adaptive_l2(u - tgt)

    sampler = None

    can_sample = (
        cfg.sample_interval > 0 and bool(cfg.sample_prompts) and diffuser is not None
    )
    global_step = 0

    for epoch in tqdm(range(start_epoch, cfg.n_epochs), desc="epoch"):
        model.train()
        dataset.set_epoch(epoch)
        total, seen = 0.0, 0

        for latent, emb in tqdm(dataloader, desc=f"epoch {epoch}", leave=False):
            optimizer.zero_grad()
            loss = step(latent, emb)
            accel.backward(loss)

            grad_norm = None
            if cfg.max_grad_norm > 0 and accel.sync_gradients:
                grad_norm = accel.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            optimizer.step()

            if ema is not None:
                ema.update(net)

            total += loss.item()
            seen += 1
            global_step += 1

            if cfg.log_interval > 0 and global_step % cfg.log_interval == 0:
                metrics = {
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }
                if grad_norm is not None:
                    metrics["grad_norm"] = grad_norm.item()
                accel.log(metrics, step=global_step)

        scheduler.step()

        mean_loss = total / max(seen, 1)
        accel.print(f"epoch {epoch}: loss {mean_loss:.4f}")
        accel.log({"epoch_loss": mean_loss}, step=global_step)

        if cfg.save_interval > 0 and (epoch + 1) % cfg.save_interval == 0:
            accel.wait_for_everyone()
            if not accel.is_main_process:
                return

            state = {
                "epoch": epoch + 1,
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            if ema is not None:
                state["ema"] = ema.state_dict()

            accel.save(
                state, os.path.join(cfg.output_dir, f"checkpoint_{epoch + 1:06d}.pt")
            )

        if can_sample and (epoch + 1) % cfg.sample_interval == 0:
            accel.wait_for_everyone()
            if not accel.is_main_process:
                return

            if sampler is None:
                sampler = Diffuser(diffuser, device=device, dit=net)

            out_dir = os.path.join(cfg.output_dir, "samples")
            os.makedirs(out_dir, exist_ok=True)

            with ema.averaged(net) if ema is not None else contextlib.nullcontext():
                images = [
                    sampler.generate(
                        prompt,
                        num_images=1,
                        num_steps=cfg.sample_steps,
                        guidance=cfg.sample_guidance,
                    )[0]
                    for prompt in cfg.sample_prompts
                ]

            paths = []
            for i, image in enumerate(images):
                path = os.path.join(out_dir, f"epoch_{epoch + 1:06d}_{i:02d}.png")
                save_image(image.float(), path)
                paths.append(path)

            tracker = None
            if cfg.wandb_project:
                with contextlib.suppress(Exception):
                    tracker = accel.get_tracker("wandb", unwrap=True)

            if tracker is not None:
                import wandb

                tracker.log(
                    {
                        "samples": [
                            wandb.Image(path, caption=prompt)
                            for path, prompt in zip(paths, cfg.sample_prompts)
                        ]
                    },
                    step=epoch + 1,
                )

    accel.wait_for_everyone()

    if accel.is_main_process:
        weights = ema.weights(net) if ema is not None else net.state_dict()
        accel.save(weights, os.path.join(cfg.output_dir, "model.pt"))

    accel.end_training()
