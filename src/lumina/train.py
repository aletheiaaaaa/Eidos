import contextlib
import dataclasses
import functools
import math
import os

import torch
import wandb
from accelerate import Accelerator
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizerFast

from .configs import DataConfig, DiffuserConfig, TrainConfig
from .data import Stream, collate
from .nn.encoder import Encoder
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

    tokenizer = CLIPTokenizerFast.from_pretrained(data.clip)

    dataset = Stream(data, seed=cfg.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        collate_fn=functools.partial(
            collate, tokenizer=tokenizer, max_tokens=data.max_tokens
        ),
    )

    def lr_lambda(step: int) -> float:
        if step < cfg.n_warmup:
            return (step + 1) / max(cfg.n_warmup, 1)
        progress = (step - cfg.n_warmup) / max(cfg.max_steps - cfg.n_warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    model, dataloader, optimizer, scheduler = accel.prepare(
        model, dataloader, optimizer, scheduler
    )

    text_encoder = (
        CLIPTextModel.from_pretrained(data.clip).to(device).eval().requires_grad_(False)
    )
    encoder = Encoder(data.encoder).to(device)

    net = accel.unwrap_model(model)
    ema = EMA(net, cfg.ema_decay) if cfg.ema_decay > 0 else None

    if cfg.compile:
        model = torch.compile(model)

    os.makedirs(cfg.output_dir, exist_ok=True)

    start_step = 0
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if "model" in state:
            net.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            if ema is not None and "ema" in state:
                ema.load_state_dict(state["ema"])
            if "stats" in state:
                encoder.load_stats(state["stats"])
            start_step = state["step"]
        else:
            net.load_state_dict(state)
        accel.print(f"resumed from {resume} at step {start_step}")

    if not bool(encoder.pixel_fitted):
        encoder.fit_stats(dataloader, data.n_stat_batches)
        pixel = [round(v, 4) for v in encoder.pixel_mean.flatten().tolist()]
        accel.print(f"pixel mean {pixel} latent std {encoder.latent_std.mean():.4f}")

    def step(x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
            lambda z, r, t: model(z, c, r, t, drop, mask),
            (z, r, t),
            (v, torch.zeros_like(r), torch.ones_like(t)),
        )

        tgt = (v - (tb - rb) * dudt).detach()

        return adaptive_l2(u - tgt)

    sampler = None

    can_sample = (
        cfg.sample_interval > 0
        and bool(cfg.sample_prompts)
        and diffuser is not None
        and bool(diffuser.decoder_path)
    )

    if cfg.sample_interval > 0 and not can_sample:
        accel.print("sampling disabled: set diffuser.decoder_path to decode latents")

    global_step = start_step
    epoch = 0

    pbar = tqdm(
        total=cfg.max_steps,
        initial=global_step,
        desc="step",
        disable=not accel.is_local_main_process,
    )

    def save_checkpoint() -> None:
        accel.wait_for_everyone()
        if not accel.is_main_process:
            return

        state = {
            "step": global_step,
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "stats": encoder.stats_dict(),
        }
        if ema is not None:
            state["ema"] = ema.state_dict()

        accel.save(
            state, os.path.join(cfg.output_dir, f"checkpoint_{global_step:08d}.pt")
        )

    def log_samples() -> None:
        nonlocal sampler

        accel.wait_for_everyone()
        if not accel.is_main_process:
            return

        if sampler is None:
            sampler = Diffuser(
                diffuser, device=device, dit=net, stats=encoder.stats_dict()
            )

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
            path = os.path.join(out_dir, f"step_{global_step:08d}_{i:02d}.png")
            save_image(image.float(), path)
            paths.append(path)

        tracker = None
        if cfg.wandb_project:
            with contextlib.suppress(Exception):
                tracker = accel.get_tracker("wandb", unwrap=True)

        if tracker is not None:
            tracker.log(
                {
                    "samples": [
                        wandb.Image(path, caption=prompt)
                        for path, prompt in zip(paths, cfg.sample_prompts)
                    ]
                },
                step=global_step,
            )

    while global_step < cfg.max_steps:
        model.train()
        dataset.set_epoch(epoch)
        total, seen = 0.0, 0

        for pixels, input_ids, attention_mask in dataloader:
            with torch.no_grad():
                latent = encoder.encode(pixels)
                emb = text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state

            optimizer.zero_grad()
            loss = step(latent, emb, attention_mask)
            accel.backward(loss)

            grad_norm = None
            if cfg.max_grad_norm > 0 and accel.sync_gradients:
                grad_norm = accel.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm
                ).item()

            optimizer.step()
            scheduler.step()

            if ema is not None:
                ema.update(net)

            value = loss.item()
            lr = scheduler.get_last_lr()[0]
            total += value
            seen += 1
            global_step += 1
            pbar.update(1)

            postfix = {
                "loss": f"{value:.4f}",
                "avg": f"{total / seen:.4f}",
                "lr": f"{lr:.2e}",
            }
            if grad_norm is not None:
                postfix["gn"] = f"{grad_norm:.2f}"
            pbar.set_postfix(postfix, refresh=False)

            if cfg.log_interval > 0 and global_step % cfg.log_interval == 0:
                metrics = {"loss": value, "lr": lr, "epoch": epoch}
                if grad_norm is not None:
                    metrics["grad_norm"] = grad_norm
                accel.log(metrics, step=global_step)

            if cfg.save_interval > 0 and global_step % cfg.save_interval == 0:
                save_checkpoint()

            if can_sample and global_step % cfg.sample_interval == 0:
                log_samples()

            if global_step >= cfg.max_steps:
                break

        accel.log({"epoch_loss": total / max(seen, 1)}, step=global_step)
        epoch += 1

    pbar.close()

    accel.wait_for_everyone()

    if accel.is_main_process:
        weights = ema.weights(net) if ema is not None else net.state_dict()
        accel.save(
            {"model": weights, "stats": encoder.stats_dict()},
            os.path.join(cfg.output_dir, "model.pt"),
        )

    accel.end_training()
