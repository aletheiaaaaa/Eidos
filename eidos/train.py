import torch
import einops
import wandb
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from .model import Diffuser
from .configs import TrainConfig
from .data import H5Dataset

accel = Accelerator(mixed_precision="bf16" if torch.cuda.is_available() else "no")
device = accel.device

def get_fm_loss(pred: torch.Tensor, target: torch.Tensor, keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    denoise = pred.gather(1, keep.unsqueeze(-1).expand(-1, -1, pred.size(-1)))
    target_patches = target.gather(1, keep.unsqueeze(-1).expand(-1, -1, target.size(-1)))

    mse = (denoise - target_patches).pow(2).mean(dim=(list(range(1, denoise.dim()))))
    cos = 1 - F.cosine_similarity(denoise.flatten(1), target_patches.flatten(1), dim=1)

    return mse + cos, target_patches

def get_cm_loss(pred: torch.Tensor, grad: torch.Tensor, keep: torch.Tensor, fm_target: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    denoise = pred.gather(1, keep.unsqueeze(-1).expand(-1, -1, pred.size(-1)))
    grad_kept = grad.gather(1, keep.unsqueeze(-1).expand(-1, -1, grad.size(-1)))

    sg = denoise.detach()
    grad_kept = grad_kept.detach()

    res = sg - (fm_target + (1 - time.reshape(-1, 1)) * grad_kept)
    alpha = 1 - time.reshape(-1, 1).pow(0.5)
    target = grad_kept - alpha * res.clamp(-1, 1)

    beta = torch.cos(time * torch.pi / 2)
    loss = (denoise - target).pow(2).mean(dim=(list(range(1, denoise.dim()))))

    return beta * loss / (loss + 1e-3).pow(0.5).detach()

def get_mae_loss(fm_pred: torch.Tensor, cm_pred: torch.Tensor, noise: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    rem_idx = torch.where(mask)

    fm_rem = fm_pred[rem_idx[0], rem_idx[1]]
    cm_rem = cm_pred[rem_idx[0], rem_idx[1]]
    noise_rem = noise[rem_idx[0], rem_idx[1]]

    mae_pred = (fm_rem + cm_rem) / 2

    return F.l1_loss(mae_pred, noise_rem, reduction="mean")

def main_loop(model: Diffuser, cfg: TrainConfig, epochs: int, lr: float, mask_freq: float, use_mae: bool, suffix: str) -> None:
    wandb.init(project="eidos", config=vars(cfg))
    model.maskdit.set_mask_freq(mask_freq)

    dataset = H5Dataset(cfg.data_path)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, drop_last=True)

    optimizer = optim.AdamW(model.maskdit.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    model.maskdit, optimizer, loader = accel.prepare(model.maskdit, optimizer, loader)
    model.maskdit.to(device)

    model.maskdit.train()
    for epoch in range(epochs):
        dataset.shard_perm()

        for latent, embed in tqdm(loader):

            noise = torch.randn_like(latent)
            sigma = (cfg.p_std * torch.randn(cfg.batch_size, device=device) + cfg.p_mean).exp()
            sigma = torch.arctan(sigma) * (2 / torch.pi)
            time = 1 - sigma

            aux = torch.ones_like(time).to(device)
            noisy_lat = time.reshape(-1, 1, 1, 1) * latent + (1 - time.reshape(-1, 1, 1, 1)) * noise
            seed = torch.randint(0, 1e6, (1,)).item()

            fm_pred_raw, keep = model.maskdit(noisy_lat, embed, time.flatten(), time.flatten(), seed=seed)
            fm_pred = einops.rearrange(fm_pred_raw, "b c (h p) (w p) -> b (h w) (p p c)", p=model.cfg.patch_size)
            fm_target = einops.rearrange(latent - noise, "b c (h p) (w p) -> b (h w) (p p c)", p=model.cfg.patch_size)

            with accel.autocast():
                fm_loss, fm_target_patches = get_fm_loss(fm_pred, fm_target, keep)

            (cm_pred, cm_grad) = torch.func.jvp(
                lambda x, t: model.maskdit(x, embed, t, aux.flatten(), seed=seed)[0],
                (noisy_lat, time.flatten()),
                (latent - noise, aux.flatten())
            )

            cm_pred = einops.rearrange(cm_pred, "b c (h p) (w p) -> b (h w) (p p c)", p=model.cfg.patch_size)
            cm_grad = einops.rearrange(cm_grad, "b c (h p) (w p) -> b (h w) (p p c)", p=model.cfg.patch_size)

            with accel.autocast():
                cm_loss = get_cm_loss(cm_pred, cm_grad, keep, fm_target_patches, time)

            loss = fm_loss.mean() + cm_loss.mean()
            logs = {"fm_loss": fm_loss.mean().item(), "cm_loss": cm_loss.mean().item()}

            if use_mae:
                all_idx = torch.arange(model.maskdit.seq_len, device=keep.device).unsqueeze(0).expand(keep.shape[0], -1)
                mask = torch.ones_like(all_idx, dtype=torch.bool).scatter_(1, keep, False)
                noise = einops.rearrange(noise, "b c (h p) (w p) -> b (h w) (p p c)", p=model.cfg.patch_size)

                with accel.autocast():
                    mae_loss = get_mae_loss(fm_pred, cm_pred, noise, mask)

                loss = loss + cfg.mae_weight * mae_loss
                logs["mae_loss"] = mae_loss.item()

            logs.update({"loss": loss.item(), "epoch": epoch})
            wandb.log(logs)

            optimizer.zero_grad()
            accel.backward(loss)
            optimizer.step()

        if epoch % cfg.save_interval == 0 or epoch == epochs - 1:
            torch.save(model.maskdit.state_dict(), f"{cfg.output_dir}/eidos_{suffix}_epoch_{epoch:04d}.pt")

@torch.compile()
def train(model: Diffuser, cfg: TrainConfig) -> None:
    main_loop(model, cfg, cfg.train_epochs, cfg.train_lr, cfg.mask_freq, True, "train")

@torch.compile()
def finetune(model: Diffuser, cfg: TrainConfig) -> None:
    main_loop(model, cfg, cfg.finetune_epochs, cfg.finetune_lr, 0.0, False, "finetune")