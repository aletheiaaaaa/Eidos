import torch
from torch import nn 
from torch.nn import functional as F
from einops import einops

class MHA(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super(MHA, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head

        self.q_proj = nn.Linear(d_model, n_heads * d_head)
        self.k_proj = nn.Linear(d_model, n_heads * d_head)
        self.v_proj = nn.Linear(d_model, n_heads * d_head)
        self.o_proj = nn.Linear(n_heads * d_head, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_pos, _ = x.size()

        queries = self.q_proj(x).view(batch, seq_pos, self.n_heads, self.d_head)
        keys = self.k_proj(x).view(batch, seq_pos, self.n_heads, self.d_head)
        values = self.v_proj(x).view(batch, seq_pos, self.n_heads, self.d_head)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = F.scaled_dot_product_attention(queries, keys, values)
        scores = scores.transpose(1, 2).contiguous().view(batch, seq_pos, self.n_heads * self.d_head)

        return self.o_proj(scores)

class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(d_model, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        return x

class Modulator(nn.Module):
    def __init__(self, d_caption, d_model):
        super(Modulator, self).__init__()

        self.alpha = nn.Linear(d_caption, d_model)
        self.beta = nn.Linear(d_caption, d_model)
        self.gamma = nn.Linear(d_caption, d_model)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.silu(x)

        alpha = self.alpha(x)
        beta = self.beta(x)
        gamma = self.gamma(x)

        return alpha, beta, gamma

class DiTBlock(nn.Module):
    def __init__(self, d_caption, d_model, n_heads, d_head, d_mlp):
        super(DiTBlock, self).__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads, d_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp)

        self.mod1 = Modulator(d_caption, d_model)
        self.mod2 = Modulator(d_caption, d_model)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        alpha1, beta1, gamma1 = self.mod1(y)
        alpha2, beta2, gamma2 = self.mod2(y)

        x = x + gamma1 * self.attn(self.ln1(x) * (1 + alpha1) + beta1)
        x = x + gamma2 * self.mlp(self.ln2(x) * (1 + alpha2) + beta2)

        return x

class FinalBlock(nn.Module):
    def __init__(self, d_caption, d_model, d_out):
        super(FinalBlock, self).__init__()

        self.ln = nn.LayerNorm(d_model)
        self.mod = Modulator(d_caption, d_model)
        self.fc = nn.Linear(d_model, d_out)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        alpha, beta, _ = self.mod(y)

        x = alpha * self.ln(x) + beta
        x = self.fc(x)

        return x

class TimeEmbed(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(start=0, end=self.d_model // 2)
        freqs = freqs / (self.d_model // 2)
        freqs = (1 / 10000) ** freqs
        x = x.outer(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)

        return x

class ImgEmbed(nn.Module):
    def __init__(self, d_model, n_channels, patch_size):
        super(ImgEmbed, self).__init__()

        self.conv = nn.Conv2d(n_channels, d_model, patch_size, patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.conv(x))

        return einops.rearrange(x, "b d h w -> b (h w) d")

class Unembed(nn.Module):
    def __init__(self, d_model, n_channels, patch_size, img_size):
        super(ImgEmbed, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size

        self.fc = nn.Linear(d_model, patch_size * patch_size * n_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc(x))
        x = einops.rearrange(
            x, 
            "b (h w) (c p q) -> b c (h p) (h q)", 
            h=self.img_size / self.patch_size, 
            p=self.patch_size, 
            q=self.patch_size
        )

        return x