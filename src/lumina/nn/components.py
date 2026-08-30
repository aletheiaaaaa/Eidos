import einops
import torch
from torch import nn
from torch.nn import functional as F


class MHA(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_head: int, d_context: int | None = None
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_context = d_model if d_context is None else d_context

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(self.d_context, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(self.d_context, n_heads * d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = x if y is None else y

        batch, seq_pos, _ = x.size()
        ctx_pos = y.size(1)

        queries = (
            self.q_proj(x)
            .view(batch, seq_pos, self.n_heads, self.d_head)
            .transpose(1, 2)
        )
        keys = (
            self.k_proj(y)
            .view(batch, ctx_pos, self.n_heads, self.d_head)
            .transpose(1, 2)
        )
        values = (
            self.v_proj(y)
            .view(batch, ctx_pos, self.n_heads, self.d_head)
            .transpose(1, 2)
        )

        weights = (queries @ keys.transpose(-1, -2)) * self.d_head**-0.5

        if mask is not None:
            weights = weights.masked_fill(
                ~mask[:, None, None, :].bool(), torch.finfo(weights.dtype).min
            )

        scores = (
            (weights.softmax(dim=-1) @ values)
            .transpose(1, 2)
            .contiguous()
            .view(batch, seq_pos, self.n_heads * self.d_head)
        )

        return self.o_proj(scores)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int) -> None:
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(d_model, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        return x


class CaptionProj(nn.Module):
    def __init__(self, d_caption: int, d_model: int) -> None:
        super(CaptionProj, self).__init__()

        self.proj = nn.Linear(d_caption, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.proj(x))


class Modulator(nn.Module):
    def __init__(self, d_caption: int, d_model: int) -> None:
        super(Modulator, self).__init__()

        self.alpha = nn.Linear(d_caption, d_model)
        self.beta = nn.Linear(d_caption, d_model)
        self.gamma = nn.Linear(d_caption, d_model)

        for mod in (self.alpha, self.beta, self.gamma):
            nn.init.zeros_(mod.weight)
            nn.init.zeros_(mod.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.silu(x)

        alpha = self.alpha(x)
        beta = self.beta(x)
        gamma = self.gamma(x)

        return alpha, beta, gamma


class DiTBlock(nn.Module):
    def __init__(
        self, d_caption: int, d_model: int, n_heads: int, d_head: int, d_mlp: int
    ) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads, d_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross = MHA(d_model, n_heads, d_head, d_context=d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp)

        self.mod1 = Modulator(d_caption, d_model)
        self.mod2 = Modulator(d_caption, d_model)
        self.mod3 = Modulator(d_caption, d_model)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        alpha1, beta1, gamma1 = self.mod1(y)
        alpha2, beta2, gamma2 = self.mod2(y)
        alpha3, beta3, gamma3 = self.mod3(y)

        x = x + gamma1 * self.attn(self.ln1(x) * (1 + alpha1) + beta1)
        x = x + gamma2 * self.cross(self.ln2(x) * (1 + alpha2) + beta2, context, mask)
        x = x + gamma3 * self.mlp(self.ln3(x) * (1 + alpha3) + beta3)

        return x


class ViTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_mlp: int) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads, d_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class FinalBlock(nn.Module):
    def __init__(self, d_caption: int, d_model: int, d_out: int) -> None:
        super(FinalBlock, self).__init__()

        self.ln = nn.LayerNorm(d_model)
        self.mod = Modulator(d_caption, d_model)
        self.fc = nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        alpha, beta, _ = self.mod(y)

        x = (1 + alpha) * self.ln(x) + beta
        x = self.fc(x)

        return x


class TimeEmbed(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(
            start=0, end=self.d_model // 2, device=t.device, dtype=t.dtype
        )
        freqs = freqs / (self.d_model // 2)
        freqs = (1 / 10000) ** freqs

        emb = None
        for x, mlp in zip([r, t], [self.mlp1, self.mlp2]):
            x = x.outer(freqs)
            x = torch.cat([x.cos(), x.sin()], dim=-1)
            x = mlp(x)

            emb = x if emb is None else emb + x

        return emb


class ImgEmbed(nn.Module):
    def __init__(self, d_model: int, n_channels: int, patch_size: int) -> None:
        super(ImgEmbed, self).__init__()

        self.conv = nn.Conv2d(n_channels, d_model, patch_size, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.conv(x))

        return einops.rearrange(x, "b c h w -> b (h w) c")


class Unembed(nn.Module):
    def __init__(
        self, d_model: int, n_channels: int, patch_size: int, img_size: int
    ) -> None:
        super(Unembed, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size

        self.fc = nn.Linear(d_model, patch_size * patch_size * n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = einops.rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=self.img_size // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
        )

        return x
