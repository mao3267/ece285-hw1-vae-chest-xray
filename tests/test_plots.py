import json
from pathlib import Path

import pytest
import torch

from vae_xray.plots import (
    plot_interp,
    plot_losses,
    plot_pca,
    plot_recon,
    plot_samples,
    write_results_table,
)
from vae_xray.train import TrainConfig, train


def _fake_run(root: Path, name: str, beta: float, fid: float, with_eval: bool = True) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"beta": beta, "name": name}))
    rows = ["epoch,split,total,recon,kl,beta,lr"]
    for epoch in (1, 2, 3):
        for split in ("train", "val"):
            rows.append(f"{epoch},{split},{100 / epoch:.2f},{90 / epoch:.2f},{10 / epoch:.2f},{beta},0.001")
    (run / "metrics.csv").write_text("\n".join(rows) + "\n")
    if with_eval:
        block = lambda f: {"fid": f, "is_mean": 1.5, "is_std": 0.1}  # noqa: E731
        (run / "eval.json").write_text(json.dumps({
            "prior": block(fid), "gmm": block(fid - 10), "recon": block(fid - 20),
            "n_samples": 2000, "n_real": 624, "device": "cpu"}))
    return run


@pytest.fixture
def trained_run(tmp_path: Path) -> tuple[Path, Path]:
    g = torch.Generator().manual_seed(0)
    cache = {}
    for split, n in (("train", 48), ("val", 16), ("test", 24)):
        cache[f"{split}_x"] = torch.randint(0, 256, (n, 1, 64, 64), dtype=torch.uint8, generator=g)
        cache[f"{split}_y"] = torch.arange(n) % 2
    cache_path = tmp_path / "cache_64.pt"
    torch.save(cache, cache_path)
    cfg = TrainConfig(name="r", epochs=1, batch_size=16, cache_path=str(cache_path),
                      device="cpu", runs_dir=str(tmp_path / "runs"), warmup_epochs=1)
    return train(cfg), cache_path


def test_write_results_table_contains_values_from_eval_json(tmp_path):
    runs = [_fake_run(tmp_path, "beta1", 1.0, 123.456), _fake_run(tmp_path, "beta0.1", 0.1, 98.76)]
    out = tmp_path / "table.tex"
    write_results_table(runs, out)
    text = out.read_text()
    assert text.count("\\\\") >= 2
    for needle in ("123.5", "113.5", "103.5", "98.8", "88.8", "78.8", "1.50", "0.10"):
        assert needle in text, needle
    assert "\\begin{tabular}" in text and "\\end{tabular}" in text


def test_write_results_table_fails_loudly_on_missing_eval(tmp_path):
    run = _fake_run(tmp_path, "beta1", 1.0, 100.0, with_eval=False)
    with pytest.raises(FileNotFoundError):
        write_results_table([run], tmp_path / "table.tex")


def test_plot_losses_writes_nonempty_file(tmp_path):
    runs = [_fake_run(tmp_path, "a", 1.0, 1.0), _fake_run(tmp_path, "b", 0.1, 1.0)]
    out = tmp_path / "losses.pdf"
    plot_losses(runs, out)
    assert out.stat().st_size > 1000


def test_image_plots_write_files(trained_run, tmp_path):
    run_dir, cache_path = trained_run
    kw = dict(device="cpu", cache_path=cache_path)
    outputs = {
        "prior.png": lambda p: plot_samples(run_dir, p, source="prior", **kw),
        "gmm.png": lambda p: plot_samples(run_dir, p, source="gmm", gmm_components=2, **kw),
        "recon.png": lambda p: plot_recon(run_dir, p, n=6, **kw),
        "interp.png": lambda p: plot_interp(run_dir, p, steps=5, **kw),
        "pca.pdf": lambda p: plot_pca(run_dir, p, **kw),
    }
    for name, fn in outputs.items():
        path = tmp_path / "figs" / name
        fn(path)
        assert path.exists() and path.stat().st_size > 500, name


def test_plot_samples_rejects_unknown_source(trained_run, tmp_path):
    run_dir, cache_path = trained_run
    with pytest.raises(ValueError):
        plot_samples(run_dir, tmp_path / "x.png", source="bogus", device="cpu",
                     cache_path=cache_path)
