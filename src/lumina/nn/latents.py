import math

import einops
import torch
from torch import nn
from transformers import AutoModel

from ..configs import DecoderConfig
from .components import Unembed, ViTBlock


def _moments(chunks, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(dim)
    total_sq = torch.zeros(dim)
    count = 0

    for flat in chunks:
        total += flat.sum(1)
        total_sq += flat.pow(2).sum(1)
        count += flat.shape[1]

    if count == 0:
        raise ValueError("no batches consumed while fitting statistics")

    mean = total / count
    std = (total_sq / count - mean.pow(2)).clamp_min(1e-12).sqrt()

    return mean, std


class Encoder(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()

        self.model = AutoModel.from_pretrained(name).eval().requires_grad_(False)
        self.patch_size = self.model.config.patch_size
        self.d_latent = self.model.config.hidden_size

        self.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.ones(1, 3, 1, 1))
        self.register_buffer("pixel_fitted", torch.zeros((), dtype=torch.bool))
        self.register_buffer("latent_mean", torch.zeros(1, self.d_latent, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, self.d_latent, 1, 1))

    def stats_dict(self) -> dict:
        return {
            key: getattr(self, key).detach().cpu().clone()
            for key in [
                "pixel_mean",
                "pixel_std",
                "pixel_fitted",
                "latent_mean",
                "latent_std",
            ]
        }

    def load_stats(self, stats: dict) -> None:
        for key in [
            "pixel_mean",
            "pixel_std",
            "pixel_fitted",
            "latent_mean",
            "latent_std",
        ]:
            getattr(self, key).copy_(stats[key])

    @torch.no_grad()
    def fit_stats(self, dataloader, n_batches: int) -> None:
        def pixels():
            for i, batch in enumerate(dataloader):
                if i >= n_batches:
                    break
                yield batch[0].float().permute(1, 0, 2, 3).reshape(3, -1).cpu()

        mean, std = _moments(pixels(), 3)
        self.pixel_mean.copy_(mean.view(1, 3, 1, 1))
        self.pixel_std.copy_(std.view(1, 3, 1, 1))
        self.pixel_fitted.fill_(True)

        def latents():
            for i, batch in enumerate(dataloader):
                if i >= n_batches:
                    break
                grid = self.encode(
                    batch[0].to(self.pixel_mean.device), normalize=False
                ).float()
                yield grid.permute(1, 0, 2, 3).reshape(self.d_latent, -1).cpu()

        mean, std = _moments(latents(), self.d_latent)
        self.latent_mean.copy_(mean.view(1, -1, 1, 1))
        self.latent_std.copy_(std.view(1, -1, 1, 1))

    def grid_size(self, resolution: int) -> int:
        return resolution // self.patch_size

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        return self.features(pixels, normalize)

    def features(self, pixels: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if not bool(self.pixel_fitted):
            raise RuntimeError(
                "statistics unfitted; call fit_stats before encoding "
                "or set data.n_stat_batches above zero"
            )

        side = self.grid_size(pixels.shape[-1])
        n_patches = side * side

        x = (pixels - self.pixel_mean) / self.pixel_std
        tokens = self.model(pixel_values=x).last_hidden_state

        if tokens.shape[1] < n_patches:
            raise ValueError(
                f"encoder returned {tokens.shape[1]} tokens, expected at least "
                f"{n_patches} for a {pixels.shape[-1]}px input"
            )

        grid = (
            tokens[:, -n_patches:]
            .transpose(1, 2)
            .reshape(pixels.shape[0], self.d_latent, side, side)
        )

        if not normalize:
            return grid

        return (grid - self.latent_mean) / self.latent_std

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.latent_mean) / self.latent_std

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.latent_std + self.latent_mean


class Decoder(nn.Module):
    def __init__(self, cfg: DecoderConfig, grid: int) -> None:
        super().__init__()

        self.grid = grid
        self.seq_len = grid * grid
        self.patch_size = cfg.resolution // grid

        if self.patch_size * grid != cfg.resolution:
            raise ValueError(
                f"resolution {cfg.resolution} is not divisible by the "
                f"{grid}x{grid} latent grid"
            )

        self.embed = nn.Linear(cfg.d_latent, cfg.d_model)
        self.pos_embed = nn.Embedding(self.seq_len, cfg.d_model)

        self.blocks = nn.ModuleList(
            [
                ViTBlock(cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_mlp)
                for _ in range(cfg.n_layers)
            ]
        )

        self.ln = nn.LayerNorm(cfg.d_model)
        self.unembed = Unembed(cfg.d_model, 3, self.patch_size, cfg.resolution)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != self.grid or latent.shape[-2] != self.grid:
            raise ValueError(
                f"decoder expects a {self.grid}x{self.grid} latent grid, "
                f"got {tuple(latent.shape[-2:])}"
            )

        x = einops.rearrange(latent, "b c h w -> b (h w) c")
        pos = self.pos_embed(torch.arange(self.seq_len, device=latent.device))

        x = self.embed(x) + pos.unsqueeze(0)

        for block in self.blocks:
            x = block(x)

        return self.unembed(self.ln(x))


class LatentDiscriminator(nn.Module):
    def __init__(self, d_latent: int, n_channels: int = 64, n_layers: int = 3) -> None:
        super().__init__()

        layers = [nn.Conv2d(d_latent, n_channels, 1), nn.LeakyReLU(0.2, inplace=True)]

        width = n_channels
        for i in range(1, n_layers + 1):
            prev, width = width, n_channels * min(2**i, 8)
            layers += [
                nn.Conv2d(prev, width, 3, 1, 1, bias=False),
                nn.GroupNorm(math.gcd(32, width), width),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        layers.append(nn.Conv2d(width, 1, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class PixelDiscriminator(nn.Module):
    def __init__(self, n_channels: int = 64, n_layers: int = 3) -> None:
        super().__init__()

        layers = [nn.Conv2d(3, n_channels, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]

        width = n_channels
        for i in range(1, n_layers + 1):
            prev, width = width, n_channels * min(2**i, 8)
            layers += [
                nn.Conv2d(prev, width, 4, 2 if i < n_layers else 1, 1, bias=False),
                nn.GroupNorm(math.gcd(32, width), width),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        layers.append(nn.Conv2d(width, 1, 4, 1, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.net(2.0 * pixels - 1.0)
