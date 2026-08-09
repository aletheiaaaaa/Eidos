import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from torch import nn
from transformers import CLIPModel, CLIPProcessor

from ..utils.configs import DiffuserConfig
from .components import CaptionProj, DiTBlock, FinalBlock, ImgEmbed, TimeEmbed, Unembed


class DiT(nn.Module):
    def __init__(self, cfg: DiffuserConfig) -> None:
        super(DiT, self).__init__()
        self.cfg = cfg
        self.seq_len = int((self.cfg.img_size / self.cfg.patch_size) ** 2)

        self.time_embed = TimeEmbed(cfg.d_caption)
        self.caption_proj = CaptionProj(cfg.d_caption, cfg.d_caption)

        self.img_embed = ImgEmbed(cfg.dit.d_model, cfg.n_channels, cfg.patch_size)
        self.pos_embed = nn.Embedding(self.seq_len, cfg.dit.d_model)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    cfg.d_caption,
                    cfg.dit.d_model,
                    cfg.dit.n_heads,
                    cfg.dit.d_head,
                    cfg.dit.d_mlp,
                )
                for _ in range(cfg.dit.n_layers)
            ]
        )

        self.final = FinalBlock(cfg.d_caption, cfg.dit.d_model, cfg.dit.d_model)
        self.unembed = Unembed(
            cfg.dit.d_model, cfg.n_channels, cfg.patch_size, cfg.img_size
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        device = x.device
        batch_size = x.size(0)

        img_emb = self.img_embed(x)
        pos_emb = (
            self.pos_embed(torch.arange(self.seq_len, device=device))
            .unsqueeze(0)
            .expand(batch_size, self.seq_len, -1)
        )
        time_emb = self.time_embed(t).unsqueeze(1).expand(batch_size, self.seq_len, -1)

        x_emb = img_emb + pos_emb
        y_proj = self.caption_proj(y).unsqueeze(1)
        cond = y_proj + time_emb

        for block in self.blocks:
            x_emb = block(x_emb, cond)
        x_emb = self.final(x_emb, cond)

        x_out = self.unembed(x_emb)

        return x_out


class Diffuser:
    def __init__(self, cfg: DiffuserConfig) -> None:
        self.cfg = cfg

        self.dit = DiT(cfg)
        self.vae = AutoencoderKL.from_pretrained(cfg.vae).eval()
        self.clip = CLIPModel.from_pretrained(cfg.clip).eval()
        self.processor = CLIPProcessor.from_pretrained(cfg.clip)

    def load_denoiser(self, path: str) -> None:
        self.dit.load_state_dict(torch.load(path))

    def generate(
        self, prompt: str, num_images: int = 4, num_steps: int = 2
    ) -> torch.Tensor:
        inputs = self.processor(text=prompt, return_tensors="pt", padding=True)
        y_in = self.clip.get_text_features(**inputs).expand(num_images, -1)

        x = torch.randn(
            (num_images, self.cfg.n_channels, self.cfg.img_size, self.cfg.img_size)
        )
        steps = torch.linspace(0, 1, num_steps + 1)

        for t_now, t_next in zip(steps[:-1], steps[1:]):
            t_in = torch.ones(x.size(0)) * t_now
            pred = self.dit(x, y_in, t_in)
            x = x + (t_next - t_now) * pred

        with torch.no_grad():
            output = (
                self.vae.decode(x * 1 / self.vae.config.scaling_factor).sample() + 1
            ) / 2

        return output
