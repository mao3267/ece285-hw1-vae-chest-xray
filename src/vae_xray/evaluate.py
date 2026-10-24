"""Evaluate a trained VAE with FID and Inception Score: `python -m vae_xray.evaluate --run beta1`."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ignite.metrics import InceptionScore
from ignite.metrics.gan.fid import fid_score
from sklearn.mixture import GaussianMixture
from torch import nn

from vae_xray.common import get_device, seed_everything
from vae_xray.data import load_cache
from vae_xray.model import VAE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IS_SPLITS = 10


class _InceptionWrapper(nn.Module):
    """torchvision InceptionV3 returning (2048-d pooled features, 1000-class logits)."""

    def __init__(self):
        super().__init__()
        from torchvision import models

        net = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.fc = net.fc
        net.fc = nn.Identity()
        self.backbone = net.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        return feats, self.fc(feats)


def load_inception(device: torch.device) -> nn.Module:
    return _InceptionWrapper().to(device).eval()


def to_inception_input(x: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) in [0,1] -> (B,3,299,299), ImageNet-normalized (as torchvision's weights expect)."""
    x = F.interpolate(x.repeat(1, 3, 1, 1), size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def inception_outputs(x: torch.Tensor, device: torch.device, batch_size: int = 64,
                      model: nn.Module | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Pooled features (N,D) and logits (N,C), computed on `device`, returned on CPU."""
    model = (model if model is not None else load_inception(device)).to(device)
    feats, logits = [], []
    for i in range(0, len(x), batch_size):
        f, l = model(to_inception_input(x[i:i + batch_size].to(device)))
        feats.append(f.cpu())
        logits.append(l.cpu())
    return torch.cat(feats), torch.cat(logits)


def _fid(real_feats: torch.Tensor, fake_feats: torch.Tensor) -> float:
    """FID via ignite's `fid_score`; float64 statistics on CPU (MPS has no float64)."""
    a, b = real_feats.double(), fake_feats.double()
    return float(fid_score(a.mean(0), b.mean(0), torch.cov(a.T), torch.cov(b.T)))


def _inception_score(logits: torch.Tensor) -> tuple[float, float]:
    """Mean and std of ignite's IS over `IS_SPLITS` equal splits of the samples."""
    scores = []
    for chunk in logits.chunk(IS_SPLITS):
        metric = InceptionScore(num_features=chunk.shape[1], feature_extractor=nn.Identity(),
                                device="cpu")
        metric.update(chunk.softmax(dim=1))
        scores.append(metric.compute())
    return float(np.mean(scores)), float(np.std(scores))


def scores_from_outputs(real_feats: torch.Tensor, fake_feats: torch.Tensor,
                        fake_logits: torch.Tensor) -> dict[str, float]:
    is_mean, is_std = _inception_score(fake_logits)
    return {"fid": _fid(real_feats, fake_feats), "is_mean": is_mean, "is_std": is_std}


def fid_and_is(real: torch.Tensor, fake: torch.Tensor, device: torch.device,
               model: nn.Module | None = None, batch_size: int = 64) -> dict[str, float]:
    """FID (fake vs real) and IS (fake) for image tensors (N,1,H,W) in [0,1]."""
    model = model if model is not None else load_inception(device)
    real_feats, _ = inception_outputs(real, device, batch_size, model)
    fake_feats, fake_logits = inception_outputs(fake, device, batch_size, model)
    return scores_from_outputs(real_feats, fake_feats, fake_logits)


def fit_latent_gmm(mu: np.ndarray, k: int = 10, seed: int = 0) -> GaussianMixture:
    return GaussianMixture(n_components=k, covariance_type="full", random_state=seed).fit(mu)


def sample_gmm(gmm: GaussianMixture, n: int) -> torch.Tensor:
    return torch.from_numpy(gmm.sample(n)[0]).float()


@torch.no_grad()
def _encode_means(model: VAE, x_uint8: torch.Tensor, device: torch.device,
                  batch_size: int = 256) -> torch.Tensor:
    out = [model.encode(x_uint8[i:i + batch_size].to(device).float() / 255.0)[0].cpu()
           for i in range(0, len(x_uint8), batch_size)]
    return torch.cat(out)


@torch.no_grad()
def _decode(model: VAE, z: torch.Tensor, device: torch.device, batch_size: int = 256) -> torch.Tensor:
    return torch.cat([model.decode(z[i:i + batch_size].to(device)).cpu()
                      for i in range(0, len(z), batch_size)])


def evaluate_run(run_dir: Path, n_samples: int = 2000, device: str | None = None,
                 inception: nn.Module | None = None, cache_path: Path | None = None,
                 gmm_components: int = 10, seed: int = 0) -> dict:
    """Score prior samples, GMM samples and test reconstructions; write run_dir/eval.json."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    dev = get_device(device)
    seed_everything(seed)

    cache = load_cache(Path(cache_path or cfg["cache_path"]))
    model = VAE(latent_dim=cfg["latent_dim"], img_size=cfg["img_size"]).to(dev)
    model.load_state_dict(torch.load(run_dir / "best.pt", weights_only=True)["model"])
    model.eval()

    inception = (inception if inception is not None else load_inception(dev)).to(dev)
    real = cache["test_x"].float() / 255.0
    real_feats, _ = inception_outputs(real, dev, model=inception)

    train_mu = _encode_means(model, cache["train_x"], dev)
    gmm = fit_latent_gmm(train_mu.numpy(), k=gmm_components, seed=seed)
    sources = {
        "prior": _decode(model, torch.randn(n_samples, cfg["latent_dim"]), dev),
        "gmm": _decode(model, sample_gmm(gmm, n_samples), dev),
        "recon": _decode(model, _encode_means(model, cache["test_x"], dev), dev),
    }
    result: dict = {}
    for name, imgs in sources.items():
        feats, logits = inception_outputs(imgs, dev, model=inception)
        result[name] = scores_from_outputs(real_feats, feats, logits)
    result.update(n_samples=n_samples, n_real=len(real), device=dev.type)
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    # Let any op missing on MPS fall back to CPU instead of crashing.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run name under --runs-dir")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--device", default=None, help="override: mps | cpu")
    args = parser.parse_args()
    result = evaluate_run(Path(args.runs_dir) / args.run, args.n_samples, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
