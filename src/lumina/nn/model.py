import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from torch import nn
from transformers import CLIPModel, CLIPProcessor

from ..configs import DiffuserConfig
from .components import CaptionProj, DiTBlock, FinalBlock, ImgEmbed, TimeEmbed, Unembed


class DiT(nn.Module):
    def __init__(self, cfg: DiffuserConfig) -> None:
        super(DiT, self).__init__()
        self.cfg = cfg
        self.seq_len = int((self.cfg.img_size / self.cfg.patch_size) ** 2)

        self.time_embed = TimeEmbed(cfg.d_caption)
        self.caption_proj = CaptionProj(cfg.d_caption, cfg.d_caption)
        self.null_caption = nn.Parameter(torch.zeros(cfg.d_caption))

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

    def drop_caption(self, c: torch.Tensor, drop: torch.Tensor | None) -> torch.Tensor:
        if drop is None:
            drop = torch.zeros(c.size(0), dtype=torch.bool, device=c.device)

        null = self.null_caption.to(c.dtype).expand_as(c)

        return torch.where(drop.unsqueeze(-1), null, c)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        drop: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = x.device

        img_emb = self.img_embed(x)
        pos_emb = self.pos_embed(torch.arange(self.seq_len, device=device)).unsqueeze(0)
        time_emb = self.time_embed(r, t).unsqueeze(1)

        x_emb = img_emb + pos_emb
        c_proj = self.caption_proj(self.drop_caption(c, drop)).unsqueeze(1)
        cond = c_proj + time_emb

        for block in self.blocks:
            x_emb = block(x_emb, cond)
        x_emb = self.final(x_emb, cond)

        x_out = self.unembed(x_emb)

        return x_out


class Diffuser:
    def __init__(
        self,
        cfg: DiffuserConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dit: DiT | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)

        self.dit = (DiT(cfg) if dit is None else dit).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(cfg.vae).to(self.device).eval()
        self.clip = CLIPModel.from_pretrained(cfg.clip).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(cfg.clip, use_fast=True)

        if dit is None and cfg.model_path:
            self.load_denoiser(cfg.model_path)

    def load_denoiser(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        for key in ("ema", "model"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break

        self.dit.load_state_dict(state)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        num_images: int = 4,
        num_steps: int = 2,
        guidance: float = 1.0,
    ) -> torch.Tensor:
        was_training = self.dit.training
        self.dit.eval()

        inputs = self.processor(text=prompt, return_tensors="pt", padding=True).to(
            self.device
        )
        y_in = self.clip.get_text_features(**inputs).pooler_output.expand(
            num_images, -1
        )

        x = torch.randn(
            (num_images, self.cfg.n_channels, self.cfg.img_size, self.cfg.img_size),
            device=self.device,
        )
        steps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)

        cfg_scale = guidance != 1.0
        if cfg_scale:
            keep = torch.zeros(num_images, dtype=torch.bool, device=self.device)
            drop = torch.cat([keep, ~keep])

        for t_now, t_next in zip(steps[:-1], steps[1:]):
            if cfg_scale:
                both = torch.cat([x, x])
                t_in = t_now.expand(both.size(0))
                r_in = t_next.expand(both.size(0))
                cond, uncond = self.dit(
                    both, torch.cat([y_in, y_in]), r_in, t_in, drop
                ).chunk(2)
                pred = uncond + guidance * (cond - uncond)
            else:
                t_in = t_now.expand(num_images)
                r_in = t_next.expand(num_images)
                pred = self.dit(x, y_in, r_in, t_in)

            x = x + (t_next - t_now) * pred

        output = (self.vae.decode(x / self.vae.config.scaling_factor).sample + 1) / 2

        if was_training:
            self.dit.train()

        return output.clamp(0, 1)
