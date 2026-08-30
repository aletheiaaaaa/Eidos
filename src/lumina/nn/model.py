import torch
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizerFast

from ..configs import DecoderConfig, DiffuserConfig
from .components import CaptionProj, DiTBlock, FinalBlock, ImgEmbed, TimeEmbed, Unembed
from .latents import Decoder


class DiT(nn.Module):
    def __init__(self, cfg: DiffuserConfig) -> None:
        super(DiT, self).__init__()
        self.cfg = cfg
        self.seq_len = int((self.cfg.img_size / self.cfg.patch_size) ** 2)

        self.time_embed = TimeEmbed(cfg.d_caption)
        self.caption_proj = CaptionProj(cfg.d_caption, cfg.dit.d_model)
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
            return c

        null = self.null_caption.to(c.dtype).expand_as(c)

        return torch.where(drop.view(-1, *([1] * (c.dim() - 1))), null, c)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        drop: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = x.device

        img_emb = self.img_embed(x)
        pos_emb = self.pos_embed(torch.arange(self.seq_len, device=device)).unsqueeze(0)

        x_emb = img_emb + pos_emb
        cond = self.time_embed(r, t).unsqueeze(1)
        context = self.caption_proj(self.drop_caption(c, drop))

        for block in self.blocks:
            x_emb = block(x_emb, cond, context, mask)
        x_emb = self.final(x_emb, cond)

        x_out = self.unembed(x_emb)

        return x_out


class Diffuser:
    def __init__(
        self,
        cfg: DiffuserConfig,
        decoder: DecoderConfig | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dit: DiT | None = None,
        model_path: str = "",
        stats: dict | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)

        self.dit = (DiT(cfg) if dit is None else dit).to(self.device)
        self.clip = CLIPTextModel.from_pretrained(cfg.clip).to(self.device).eval()
        self.tokenizer = CLIPTokenizerFast.from_pretrained(cfg.clip)

        self.stats = None
        self.latent_mean, self.latent_std = None, None
        if stats is not None:
            self.load_stats(stats)

        if dit is None and model_path:
            self.load_denoiser(model_path)

        self.decoder = None
        if decoder is not None and decoder.path:
            self.decoder = (
                Decoder(decoder, cfg.img_size)
                .to(self.device)
                .eval()
                .requires_grad_(False)
            )
            self.load_decoder(decoder.path)

    def load_decoder(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)

        if isinstance(state, dict) and "stats" in state:
            self.check_stats(state["stats"], path)

        for key in ("ema", "model"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break

        self.decoder.load_state_dict(state)

    def check_stats(self, stats: dict, path: str) -> None:
        if self.stats is None:
            return

        for key in ("pixel_mean", "pixel_std"):
            a, b = self.stats[key].cpu(), stats[key].cpu()
            if not torch.allclose(a, b, atol=1e-3):
                print(
                    f"warning: {path} was trained with {key} {b.flatten().tolist()} "
                    f"but the denoiser expects {a.flatten().tolist()}; "
                    "the two saw different encoders"
                )

    def load_stats(self, stats: dict) -> None:
        self.stats = stats
        self.latent_mean = stats["latent_mean"].to(self.device)
        self.latent_std = stats["latent_std"].to(self.device)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError(
                "no decoder: diffusion runs in DINO latent space, so pixels require "
                "a trained decoder at decoder.path"
            )

        if self.latent_std is not None:
            latent = latent * self.latent_std + self.latent_mean

        return self.decoder(latent)

    def load_denoiser(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)

        if isinstance(state, dict) and "stats" in state:
            self.load_stats(state["stats"])

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

        inputs = self.tokenizer(
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).to(self.device)

        y_in = self.clip(**inputs).last_hidden_state.expand(num_images, -1, -1)
        mask_in = inputs["attention_mask"].expand(num_images, -1)

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
                    both,
                    torch.cat([y_in, y_in]),
                    r_in,
                    t_in,
                    drop,
                    torch.cat([mask_in, mask_in]),
                ).chunk(2)
                pred = uncond + guidance * (cond - uncond)
            else:
                t_in = t_now.expand(num_images)
                r_in = t_next.expand(num_images)
                pred = self.dit(x, y_in, r_in, t_in, None, mask_in)

            x = x + (t_next - t_now) * pred

        output = self.decode(x)

        if was_training:
            self.dit.train()

        return output.clamp(0, 1)
