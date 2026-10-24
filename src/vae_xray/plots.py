"""Report figures and the results table: `python -m vae_xray.plots --runs beta1 beta0.1 --best beta1`."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from torchvision.utils import save_image  # noqa: E402

from vae_xray.common import get_device, seed_everything  # noqa: E402
from vae_xray.data import load_cache  # noqa: E402
from vae_xray.evaluate import (  # noqa: E402
    _decode,
    _encode_means,
    fit_latent_gmm,
    load_run_model,
    sample_gmm,
)

CLASS_NAMES = ("NORMAL", "PNEUMONIA")


def _beta(run_dir: Path) -> float:
    return json.loads((Path(run_dir) / "config.json").read_text())["beta"]


def _setup(run_dir: Path, device: str | None, cache_path: Path | None):
    dev = get_device(device)
    model, cfg = load_run_model(run_dir, dev)
    cache = load_cache(Path(cache_path or cfg["cache_path"]))
    return dev, model, cfg, cache


def plot_losses(run_dirs: list[Path], out: Path, min_epoch: int = 5) -> None:
    """Total / reconstruction / KL per epoch; one color per run, train solid, val dashed.

    Epochs before `min_epoch` are left out: the KL warm-up makes the first epochs' values
    orders of magnitude larger and would flatten everything after them.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.45))  # near the printed width: fonts stay readable
    for color, run_dir in zip(plt.get_cmap("tab10").colors, run_dirs):
        df = pd.read_csv(Path(run_dir) / "metrics.csv")
        df = df[df.epoch >= min_epoch]
        label = rf"$\beta$={_beta(run_dir):g}"
        for ax, col in zip(axes, ("total", "recon", "kl")):
            for split, style in (("train", "-"), ("val", "--")):
                d = df[df.split == split]
                ax.plot(d.epoch, d[col], style, color=color, lw=1.2,
                        label=f"{label} {split}" if col == "total" else None)
    for ax, title in zip(axes, ("Total loss", "Reconstruction (BCE)", "KL divergence")):
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("epoch", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=4.5, ncol=2, handlelength=1.5, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_samples(run_dir: Path, out: Path, source: str = "prior", n: int = 64,
                 device: str | None = None, cache_path: Path | None = None,
                 gmm_components: int = 10, seed: int = 0) -> None:
    """8x8 grid of generated images from N(0, I) ('prior') or a latent GMM ('gmm')."""
    if source not in ("prior", "gmm"):
        raise ValueError(f"source must be 'prior' or 'gmm', got {source!r}")
    dev, model, cfg, cache = _setup(run_dir, device, cache_path)
    seed_everything(seed)
    if source == "prior":
        z = torch.randn(n, cfg["latent_dim"])
    else:
        mu = _encode_means(model, cache["train_x"], dev)
        z = sample_gmm(fit_latent_gmm(mu.numpy(), k=gmm_components, seed=seed), n)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_image(_decode(model, z, dev), out, nrow=8, padding=2)


def plot_recon(run_dir: Path, out: Path, n: int = 8, device: str | None = None,
               cache_path: Path | None = None) -> None:
    """Top row: real test images; bottom row: their reconstructions."""
    dev, model, _, cache = _setup(run_dir, device, cache_path)
    real = cache["test_x"][:n]
    recon = _decode(model, _encode_means(model, real, dev), dev)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.cat([real.float() / 255.0, recon]), out, nrow=n, padding=2)


def plot_interp(run_dir: Path, out: Path, steps: int = 10, device: str | None = None,
                cache_path: Path | None = None) -> None:
    """Decode a straight line in latent space between a NORMAL and a PNEUMONIA test image."""
    dev, model, _, cache = _setup(run_dir, device, cache_path)
    y = cache["test_y"]
    i, j = int((y == 0).nonzero()[0]), int((y == 1).nonzero()[0])
    mu = _encode_means(model, cache["test_x"][[i, j]], dev)
    t = torch.linspace(0, 1, steps).unsqueeze(1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_image(_decode(model, (1 - t) * mu[0] + t * mu[1], dev), out, nrow=steps, padding=2)


def plot_pca(run_dir: Path, out: Path, device: str | None = None,
             cache_path: Path | None = None) -> None:
    """PCA of test-set latent means, colored by class (only to look; claims need the plot)."""
    dev, model, _, cache = _setup(run_dir, device, cache_path)
    mu = _encode_means(model, cache["test_x"], dev).numpy()
    xy = PCA(n_components=2, random_state=0).fit_transform(mu)
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    for label, name in enumerate(CLASS_NAMES):
        m = (cache["test_y"] == label).numpy()
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.6, label=name)
    ax.set_xlabel("PC 1", fontsize=8)
    ax.set_ylabel("PC 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def write_results_table(run_dirs: list[Path], out: Path) -> None:
    """LaTeX tabular, one row per run: FID of prior, GMM and GMM@624 (as many generated images as
    real ones, so it is comparable with recon, which has 624 images) and recon, plus prior IS."""
    rows = []
    for run_dir in run_dirs:
        eval_path = Path(run_dir) / "eval.json"
        if not eval_path.exists():
            raise FileNotFoundError(f"{eval_path} missing; run evaluate first")
        e = json.loads(eval_path.read_text())
        rows.append(
            f"{_beta(run_dir):g} & {e['prior']['fid']:.1f} & {e['gmm']['fid']:.1f} & "
            f"{e['gmm']['fid_matched']:.1f} & {e['recon']['fid']:.1f} & "
            f"{e['prior']['is_mean']:.2f} $\\pm$ "
            f"{e['prior']['is_std']:.2f} \\\\"
        )
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "$\\beta$ & FID prior & FID GMM & FID GMM@624 & FID recon. & IS prior \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="run names under --runs-dir")
    parser.add_argument("--best", required=True, help="run used for the image figures")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="report/figures")
    parser.add_argument("--table", default="report/results_table.tex")
    parser.add_argument("--device", default=None, help="override: mps | cpu")
    args = parser.parse_args()

    runs = [Path(args.runs_dir) / r for r in args.runs]
    best, out = Path(args.runs_dir) / args.best, Path(args.out_dir)
    plot_losses(runs, out / "losses.pdf")
    plot_samples(best, out / "samples_prior.png", "prior", device=args.device)
    plot_samples(best, out / "samples_gmm.png", "gmm", device=args.device)
    plot_recon(best, out / "recon.png", device=args.device)
    plot_interp(best, out / "interp.png", device=args.device)
    plot_pca(best, out / "pca.pdf", device=args.device)
    write_results_table(runs, Path(args.table))
    print(f"wrote figures to {out} and table to {args.table}")


if __name__ == "__main__":
    main()
