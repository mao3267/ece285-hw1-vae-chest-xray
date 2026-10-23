import math

import pytest
import torch
import torch.nn.functional as F

from vae_xray.model import VAE, beta_at_epoch, vae_loss


def test_forward_shapes():
    model = VAE()
    x_hat, mu, logvar = model(torch.rand(4, 1, 64, 64))
    assert x_hat.shape == (4, 1, 64, 64)
    assert mu.shape == (4, 128) and logvar.shape == (4, 128)


def test_decode_output_in_unit_range():
    out = VAE().decode(torch.randn(5, 128) * 10)
    assert out.shape == (5, 1, 64, 64)
    assert out.min() >= 0 and out.max() <= 1


def test_sample_shape():
    assert VAE().sample(3, torch.device("cpu")).shape == (3, 1, 64, 64)


def test_kl_is_zero_for_standard_normal_posterior():
    x = torch.rand(4, 1, 64, 64)
    out = vae_loss(x, torch.rand_like(x), torch.zeros(4, 128), torch.zeros(4, 128), beta=1.0)
    assert out["kl"].abs() < 1e-6


def test_kl_matches_closed_form():
    mu, logvar = torch.randn(6, 128), torch.randn(6, 128)
    x = torch.rand(6, 1, 64, 64)
    out = vae_loss(x, torch.rand_like(x), mu, logvar, beta=1.0)
    expected = (-0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(dim=1)).mean()
    assert torch.allclose(out["kl"], expected, rtol=1e-5)


def test_total_equals_recon_plus_beta_kl():
    mu, logvar = torch.randn(3, 128), torch.randn(3, 128)
    x = torch.rand(3, 1, 64, 64)
    out = vae_loss(x, torch.rand_like(x), mu, logvar, beta=0.1)
    assert torch.allclose(out["total"], out["recon"] + 0.1 * out["kl"])


def test_loss_finite_for_extreme_logvar():
    model = VAE()
    # Force the logvar head to emit huge values; encode must clamp them.
    with torch.no_grad():
        model.fc_logvar.bias.fill_(50.0)
    x = torch.rand(2, 1, 64, 64)
    x_hat, mu, logvar = model(x)
    out = vae_loss(x, x_hat, mu, logvar, beta=1.0)
    assert all(torch.isfinite(v) for v in out.values())
    assert logvar.max() <= 10.0 and logvar.min() >= -10.0


def test_beta_schedule():
    assert beta_at_epoch(0, 1.0) == 0
    assert beta_at_epoch(5, 1.0) == pytest.approx(0.5)
    assert beta_at_epoch(10, 1.0) == pytest.approx(1.0)
    assert beta_at_epoch(50, 0.1) == pytest.approx(0.1)


def test_overfit_smoke():
    torch.manual_seed(0)
    # Structured images (smooth blobs) so there is something to learn.
    base = torch.rand(64, 1, 8, 8)
    x = torch.nn.functional.interpolate(base, size=64, mode="bilinear", align_corners=False)
    model = VAE(latent_dim=16, widths=(8, 16, 32, 64))  # small: keep the suite fast
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # BCE against soft targets cannot go below the data's own entropy; measure progress above it.
    floor = F.binary_cross_entropy(x, x, reduction="sum").item() / x.shape[0]
    first = last = None
    for step in range(150):
        x_hat, mu, logvar = model(x)
        out = vae_loss(x, x_hat, mu, logvar, beta=1.0)
        opt.zero_grad()
        out["total"].backward()
        opt.step()
        if step == 0:
            first = out["recon"].item()
        last, kl = out["recon"].item(), out["kl"].item()
    assert math.isfinite(last)
    assert (last - floor) < 0.6 * (first - floor), (floor, first, last)
    assert kl > 1.0, kl
