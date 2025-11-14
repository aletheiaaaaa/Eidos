import torch
from torch import nn
from torch.nn import functional as F
from diffusers import AutoencoderKL
from transformers import CLIPModel, CLIPProcessor

from components import ImgEmbed, TimeEmbed, DiTBlock, FinalBlock, Unembed, CaptionProj
from configs import DiffuserConfig

class MaskDiT(nn.Module):
    def __init__(self, cfg: DiffuserConfig) -> None:
        super(MaskDiT, self).__init__()
        self.cfg = cfg
        self.seq_len = int((self.cfg.img_size / self.cfg.patch_size) ** 2)

        self.time_embed = TimeEmbed(cfg.encoder.d_model)
        self.end_embed = TimeEmbed(cfg.encoder.d_model)
        self.caption_proj = CaptionProj(cfg.d_caption, cfg.d_caption)

        self.img_embed = ImgEmbed(cfg.encoder.d_model, cfg.n_channels, cfg.patch_size)
        self.pos_embed = nn.Embedding(self.seq_len, cfg.encoder.d_model)
        self.dec_embed = nn.Embedding(self.seq_len, cfg.decoder.d_model)

        self.mask_vector = nn.Parameter(torch.randn(cfg.decoder.d_model))
        self.encoder = nn.ModuleList([
            DiTBlock(cfg.d_caption, cfg.encoder.d_model, cfg.encoder.n_heads, cfg.encoder.d_head, cfg.encoder.d_mlp) for _ in range(cfg.encoder.n_layers)
        ])
        self.decoder = nn.ModuleList([
            DiTBlock(cfg.d_caption, cfg.decoder.d_model, cfg.decoder.n_heads, cfg.decoder.d_head, cfg.decoder.d_mlp) for _ in range(cfg.decoder.n_layers)
        ])

        self.enc_final = FinalBlock(cfg.d_caption, cfg.encoder.d_model, cfg.decoder.d_model)
        self.dec_final = FinalBlock(cfg.d_caption, cfg.decoder.d_model, cfg.decoder.d_model)
        self.unembed = Unembed(cfg.decoder.d_model, cfg.n_channels, cfg.patch_size, cfg.img_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch_size = x.size(0)

        img_emb = self.img_embed(x)
        pos_emb = self.pos_embed(torch.arange(self.seq_len, device=device)).unsqueeze(0).expand(batch_size, self.seq_len, -1)
        time_emb = self.time_embed(t).unsqueeze(1).expand(batch_size, self.seq_len, -1)
        end_emb = self.end_embed(r).unsqueeze(1).expand(batch_size, self.seq_len, -1)

        x_emb = img_emb + pos_emb
        y_proj = self.caption_proj(y)
        cond = y_proj + time_emb + end_emb

        num_masked = int(self.seq_len * self.cfg.mask_freq)
        idx = torch.randint(0, self.seq_len, (batch_size, num_masked), device=device)
        x_enc = torch.gather(x_emb, 1, idx.unsqueeze(-1).expand(-1, -1, x_emb.size(-1)))

        for encoder in self.encoder:
            x_enc = encoder(x_enc, cond)
        x_enc = self.enc_final(x_enc)

        x_dec = self.mask_vector.unsqueeze(0).unsqueeze(0).expand(batch_size, self.seq_len, -1)
        x_dec = x_dec.scatter(1, idx.unsqueeze(-1).expand(-1, -1, x_enc.size(-1)), x_enc)
        x_dec = x_dec + self.dec_embed(torch.arange(self.seq_len, device=device)).unsqueeze(0).expand(batch_size, self.seq_len, -1)

        for decoder in self.decoder:
            x_dec = decoder(x_dec, cond)
        x_dec = self.dec_final(x_dec)

        x_out = self.unembed(x_dec)

        return x_out

class Diffuser:
    def __init__(self, cfg: DiffuserConfig, device: torch.device) -> None:
        self.device = device

        self.cfg = cfg

        self.maskdit = MaskDiT(cfg).to(device)
        self.vae = AutoencoderKL.from_pretrained(cfg.vae).to(device).eval()
        self.clip = CLIPModel.from_pretrained(cfg.clip).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(cfg.clip)

    def load_denoiser(self, path: str) -> None:
        self.maskdit.load_state_dict(torch.load(path, map_location=self.device))

    def generate(self, prompt: str, num_images: int = 4, num_steps: int = 2) -> torch.Tensor:
        inputs = self.processor(text=prompt, return_tensors="pt", padding=True).to(self.device)
        y_in = self.clip.get_text_features(**inputs).expand(num_images, -1)

        x_next = torch.randn((num_images, self.cfg.n_channels, self.cfg.img_size, self.cfg.img_size), device=self.device)
        steps = torch.linspace(0, 1, num_steps + 1, device=self.device)

        for (_, (t_now, t_next)) in enumerate(zip(steps[:-1], steps[1:])):
            x_now = x_next

            t_in = torch.ones(x_now.size(0), device=self.device) * t_now
            r_in = torch.ones(x_now.size(0), device=self.device)

            pred = self.maskdit(x_now, y_in, t_in, r_in)

            x_end = x_now + (steps[-1] - t_now) * pred
            noise = torch.rand_like(x_end)
            x_next = t_next * x_end + (1 - t_next) * noise
        
        with torch.no_grad():
            output = (self.vae.decode(x_end * 1 / self.vae.config.scaling_factor).sample() + 1) / 2

        return output


