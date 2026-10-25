"""Convolutional VAE, its loss, and the KL-weight schedule."""

from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn

LOGVAR_CLAMP = 10.0


def _enc_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(0.2),
    )


def _dec_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(0.2),
    )


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 128, widths: tuple[int, ...] = (64, 128, 256, 512),
                 img_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feat_size = img_size // 2 ** len(widths)  # spatial size after the encoder
        self.feat_ch = widths[-1]
        flat = self.feat_ch * self.feat_size**2

        chans = (1, *widths)
        self.encoder = nn.Sequential(*[_enc_block(a, b) for a, b in pairwise(chans)])
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

        rev = widths[::-1]
        self.fc_dec = nn.Linear(latent_dim, flat)
        self.decoder = nn.Sequential(
            *[_dec_block(a, b) for a, b in pairwise(rev)],
            nn.ConvTranspose2d(rev[-1], 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-LOGVAR_CLAMP, LOGVAR_CLAMP)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, self.feat_ch, self.feat_size, self.feat_size)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        return self.decode(torch.randn(n, self.latent_dim, device=device))


def vae_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
             beta: float) -> dict[str, torch.Tensor]:
    """BCE summed over pixels + beta * closed-form KL(q(z|x) || N(0, I)), batch-averaged."""
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.shape[0]
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    return {"total": recon + beta * kl, "recon": recon, "kl": kl}


def beta_at_epoch(epoch: int, target: float, warmup_epochs: int = 10) -> float:
    """Linear KL warm-up from 0 to `target` over `warmup_epochs` (epoch is 0-indexed)."""
    return target * min(1.0, epoch / warmup_epochs)
