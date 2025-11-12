import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from components import ImgEmbed, TimeEmbed, DiTBlock, FinalBlock, Unembed
from diffusers import AutoencoderKL
from transformers import CLIPModel

class MaskDiT(nn.Module):
    def __init__(self, cfg):
        super(MaskDiT, self).__init__()
        self.cfg = cfg
        self.seq_len = int((self.cfg.img_size / self.cfg.patch_size) ** 2)

        self.time_embed = TimeEmbed(cfg.encoder.d_model)
        self.img_embed = ImgEmbed(cfg.encoder.d_model, cfg.n_channels, cfg.patch_size)
        self.pos_embed = nn.Embedding(self.seq_len, cfg.encoder.d_model)
        self.dec_embed = nn.Embedding(self.seq_len, cfg.decoder.d_model)

        self.mask_vector = nn.Parameter(torch.randn(cfg.decoder.d_model))
        self.encoder = nn.ModuleList([
            DiTBlock(cfg.encoder.d_model, cfg.encoder.n_heads, cfg.encoder.d_head, cfg.encoder.d_mlp) for _ in range(cfg.encoder.n_layers)
        ])
        self.decoder = nn.ModuleList([
            DiTBlock(cfg.decoder.d_model, cfg.decoder.n_heads, cfg.decoder.d_head, cfg.decoder.d_mlp) for _ in range(cfg.decoder.n_layers)
        ])

        self.enc_final = FinalBlock(cfg.encoder.d_model, cfg.decoder.d_model)
        self.dec_final = FinalBlock(cfg.decoder.d_model, cfg.decoder.d_model)
        self.unembed = Unembed(cfg.decoder.d_model, cfg.n_channels, cfg.patch_size, cfg.img_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch_size = x.size(0)

        img_emb = self.img_embed(x)
        pos_emb = self.pos_embed(torch.arange(self.seq_len, device=device)).unsqueeze(0).expand(batch_size, self.seq_len, -1)
        time_emb = self.time_embed(t).unsqueeze(1).expand(batch_size, self.seq_len, -1)

        x = img_emb + pos_emb + time_emb

        num_masked = int(self.seq_len * self.cfg.mask_freq)
        idx = torch.randint(0, self.seq_len, (batch_size, num_masked), device=device)
        x_enc = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))

        for encoder in self.encoder:
            x_enc = encoder(x_enc, y)
        x_enc = self.enc_final(x_enc)

        x_dec = self.mask_vector.unsqueeze(0).unsqueeze(0).expand(batch_size, self.seq_len, -1)
        x_dec = x_dec.scatter(1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)), x_enc)
        x_dec = x_dec + self.dec_embed(torch.arange(self.seq_len, device=device)).unsqueeze(0).expand(batch_size, self.seq_len, -1)

        for decoder in self.decoder:
            x_dec = decoder(x_dec, y)
        x_dec = self.dec_final(x_dec)

        x_out = self.unembed(x_dec)

        return x_out

class Weighting(nn.Module):
    def __init__(self, cfg):
        super(Weighting, self).__init__()

        self.d_model = cfg.d_model_wt

        self.fc1 = nn.Linear(1, self.d_model)
        self.fc2 = nn.Linear(self.d_model, self.d_model)
        self.fc3 = nn.Linear(self.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)

        return x

class Diffuser:
    def __init__(self, cfg, device):
        self.device = device

        self.weighting = Weighting(cfg).to(device)
        self.maskdit = MaskDiT(cfg).to(device)
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    def generate(self, prompt: str, num_images: int = 4, num_steps: int = 2):
        label = self.clip.encode_text(self.clip.tokenize(prompt, truncate=True).to(self.device))
        
