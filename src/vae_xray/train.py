"""Training loop and CLI: `uv run python -m vae_xray.train --name beta1 --beta 1.0`."""

import argparse
import csv
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torchvision.utils import save_image

from vae_xray.common import get_device, seed_everything
from vae_xray.data import load_cache, make_loaders
from vae_xray.model import VAE, beta_at_epoch, vae_loss

METRIC_FIELDS = ["epoch", "split", "total", "recon", "kl", "beta", "lr"]


@dataclass
class TrainConfig:
    name: str
    beta: float = 1.0
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    latent_dim: int = 128
    img_size: int = 64
    warmup_epochs: int = 10
    seed: int = 0
    cache_path: str = "data/cache_64.pt"
    device: str | None = None
    runs_dir: str = "runs"
    sample_every: int = 10


def _state(model, optimizer, epoch: int, cfg: TrainConfig) -> dict:
    return {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": asdict(cfg),
    }


def _run_epoch(model, loader, device, beta: float, optimizer=None) -> dict[str, float]:
    """One pass over `loader`; trains when an optimizer is given. Returns per-sample means."""
    model.train(optimizer is not None)
    sums = {"total": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, _ in loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            out = vae_loss(x, x_hat, mu, logvar, beta)
            if optimizer is not None:
                optimizer.zero_grad()
                out["total"].backward()
                optimizer.step()
            for k in sums:
                sums[k] += out[k].item() * x.shape[0]
            n += x.shape[0]
    return {k: v / n for k, v in sums.items()}


def train(cfg: TrainConfig, force: bool = False,
          epoch_hook: Callable[[int], None] | None = None) -> Path:
    """Train one VAE and return its run directory.

    `epoch_hook(epoch)` runs after each finished epoch (1-based), once that epoch's metrics
    and `last.pt` are on disk; raising KeyboardInterrupt from it stops training cleanly.
    """
    run_dir = Path(cfg.runs_dir) / cfg.name
    if run_dir.exists():
        if not force:
            raise FileExistsError(f"{run_dir} exists; pass force=True / --force to overwrite")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    device = get_device(cfg.device)
    seed_everything(cfg.seed)
    (run_dir / "config.json").write_text(
        json.dumps({**asdict(cfg), "device_used": device.type}, indent=2))

    train_loader, val_loader = make_loaders(load_cache(Path(cfg.cache_path)), cfg.batch_size)
    model = VAE(latent_dim=cfg.latent_dim, img_size=cfg.img_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    fixed_noise = torch.randn(64, cfg.latent_dim, generator=torch.Generator().manual_seed(cfg.seed))

    best_val = float("inf")
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        try:
            for epoch in range(1, cfg.epochs + 1):
                t0 = time.time()
                beta_t = beta_at_epoch(epoch - 1, cfg.beta, cfg.warmup_epochs)
                lr = optimizer.param_groups[0]["lr"]
                tr = _run_epoch(model, train_loader, device, beta_t, optimizer)
                va = _run_epoch(model, val_loader, device, cfg.beta)  # val at target beta
                scheduler.step(va["total"])

                writer.writerow({"epoch": epoch, "split": "train", **tr, "beta": beta_t, "lr": lr})
                writer.writerow({"epoch": epoch, "split": "val", **va, "beta": cfg.beta, "lr": lr})
                f.flush()

                state = _state(model, optimizer, epoch, cfg)
                torch.save(state, run_dir / "last.pt")
                if va["total"] < best_val:
                    best_val = va["total"]
                    torch.save(state, run_dir / "best.pt")
                if epoch % cfg.sample_every == 0:
                    model.eval()
                    with torch.no_grad():
                        imgs = model.decode(fixed_noise.to(device)).cpu()
                    save_image(imgs, run_dir / f"samples_epoch{epoch}.png", nrow=8)

                print(f"[{cfg.name}] epoch {epoch}/{cfg.epochs} train {tr['total']:.1f} "
                      f"val {va['total']:.1f} (recon {va['recon']:.1f} kl {va['kl']:.1f}) "
                      f"{time.time() - t0:.1f}s on {device.type}", flush=True)
                if epoch_hook is not None:
                    epoch_hook(epoch)
        except KeyboardInterrupt:
            print(f"[{cfg.name}] interrupted; last.pt holds the last finished epoch", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig(name="")
    for fld in fields(TrainConfig):
        flag = "--" + fld.name.replace("_", "-")
        if fld.name == "name":
            parser.add_argument(flag, required=True)
        elif fld.name == "device":
            parser.add_argument(flag, default=None, help="override: mps | cpu")
        else:
            parser.add_argument(flag, type=type(getattr(defaults, fld.name)),
                                default=getattr(defaults, fld.name))
    parser.add_argument("--force", action="store_true", help="overwrite an existing run")
    args = vars(parser.parse_args())
    force = args.pop("force")
    train(TrainConfig(**args), force=force)


if __name__ == "__main__":
    main()
