import torch
from torch import nn
from transformers import AutoModel

STAT_KEYS = ("pixel_mean", "pixel_std", "pixel_fitted", "latent_mean", "latent_std")


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


class DinoEncoder(nn.Module):
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
        return {key: getattr(self, key).detach().cpu().clone() for key in STAT_KEYS}

    def load_stats(self, stats: dict) -> None:
        for key in STAT_KEYS:
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

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.latent_std + self.latent_mean
