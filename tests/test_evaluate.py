import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ignite.metrics import FID

from vae_xray.evaluate import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    evaluate_run,
    fid_and_is,
    fit_latent_gmm,
    inception_outputs,
    sample_gmm,
    to_inception_input,
)
from vae_xray.train import TrainConfig, train

CPU = torch.device("cpu")


class FakeInception(torch.nn.Module):
    """Offline stand-in for InceptionV3: (B,3,H,W) -> (16-d features, 10-class logits)."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.w_feat = torch.randn(3 * 4, 16, generator=g)
        self.w_logit = torch.randn(16, 10, generator=g)

    def forward(self, x):
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, 2).flatten(1)  # (B, 12)
        feats = pooled @ self.w_feat
        return feats, feats @ self.w_logit


def _images(n: int, seed: int, shift: float = 0.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 1, 64, 64, generator=g) * 0.5 + shift).clamp(0, 1)


def test_to_inception_input_shape_and_channels():
    out = to_inception_input(_images(2, 0))
    assert out.shape == (2, 3, 299, 299)
    # Undo the per-channel ImageNet normalization: the channels must be replicas of one gray image.
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    raw = out * std + mean
    assert torch.allclose(raw[:, 0], raw[:, 1], atol=1e-5)
    assert torch.allclose(raw[:, 0], raw[:, 2], atol=1e-5)


def test_inception_outputs_batches_and_returns_cpu():
    feats, logits = inception_outputs(_images(10, 0), CPU, batch_size=4, model=FakeInception())
    assert feats.shape == (10, 16) and logits.shape == (10, 10)
    assert feats.device.type == "cpu"


def test_fid_of_identical_sets_is_near_zero():
    x = _images(60, 1)
    scores = fid_and_is(x, x, CPU, model=FakeInception())
    assert scores["fid"] < 1e-3


def test_fid_grows_with_distribution_shift():
    real = _images(80, 1)
    same = fid_and_is(real, _images(80, 2), CPU, model=FakeInception())["fid"]
    shifted = fid_and_is(real, _images(80, 2, shift=0.4), CPU, model=FakeInception())["fid"]
    assert shifted > same


def test_fid_matches_ignite_metric_when_sizes_are_equal():
    real, fake = _images(64, 3), _images(64, 4, shift=0.2)
    model = FakeInception()
    ours = fid_and_is(real, fake, CPU, model=model)["fid"]
    f_real, _ = inception_outputs(real, CPU, model=model)
    f_fake, _ = inception_outputs(fake, CPU, model=model)
    metric = FID(num_features=16, feature_extractor=torch.nn.Identity(), device="cpu")
    metric.update((f_fake, f_real))
    assert ours == pytest.approx(metric.compute(), rel=1e-4)


def test_fid_accepts_different_set_sizes():
    scores = fid_and_is(_images(40, 5), _images(120, 6), CPU, model=FakeInception())
    assert np.isfinite(scores["fid"])


def test_metric_values_are_python_floats():
    scores = fid_and_is(_images(40, 5), _images(40, 6), CPU, model=FakeInception())
    assert set(scores) == {"fid", "is_mean", "is_std"}
    assert all(type(v) is float for v in scores.values())
    assert scores["is_mean"] >= 1.0


def test_fit_and_sample_gmm_shapes():
    mu = np.random.default_rng(0).normal(size=(500, 16))
    gmm = fit_latent_gmm(mu, k=3, seed=0)
    z = sample_gmm(gmm, 40)
    assert z.shape == (40, 16) and z.dtype == torch.float32


def test_evaluate_run_writes_eval_json(tmp_path: Path):
    g = torch.Generator().manual_seed(0)
    cache = {
        f"{s}_{k}": v
        for s, n in (("train", 48), ("val", 16), ("test", 24))
        for k, v in (("x", torch.randint(0, 256, (n, 1, 64, 64), dtype=torch.uint8, generator=g)),
                     ("y", torch.zeros(n, dtype=torch.int64)))
    }
    cache_path = tmp_path / "cache_64.pt"
    torch.save(cache, cache_path)
    cfg = TrainConfig(name="r", epochs=1, batch_size=16, cache_path=str(cache_path),
                      device="cpu", runs_dir=str(tmp_path / "runs"), warmup_epochs=1)
    run_dir = train(cfg)

    result = evaluate_run(run_dir, n_samples=30, device="cpu", inception=FakeInception(),
                          cache_path=cache_path, gmm_components=2)
    saved = json.loads((run_dir / "eval.json").read_text())
    assert saved == result
    for key in ("prior", "gmm", "recon"):
        assert set(saved[key]) == {"fid", "is_mean", "is_std"}
        assert np.isfinite(saved[key]["fid"])
    assert saved["n_samples"] == 30 and saved["n_real"] == 24 and saved["device"] == "cpu"
