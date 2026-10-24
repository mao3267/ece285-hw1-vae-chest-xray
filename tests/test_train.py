import csv
import json
from pathlib import Path

import pytest
import torch

from vae_xray import train as train_module
from vae_xray.train import TrainConfig, run_finished, train


@pytest.fixture
def tiny_cache(tmp_path: Path) -> str:
    g = torch.Generator().manual_seed(0)
    cache = {
        "train_x": torch.randint(0, 256, (64, 1, 64, 64), dtype=torch.uint8, generator=g),
        "train_y": torch.zeros(64, dtype=torch.int64),
        "val_x": torch.randint(0, 256, (16, 1, 64, 64), dtype=torch.uint8, generator=g),
        "val_y": torch.zeros(16, dtype=torch.int64),
        "test_x": torch.randint(0, 256, (16, 1, 64, 64), dtype=torch.uint8, generator=g),
        "test_y": torch.zeros(16, dtype=torch.int64),
    }
    path = tmp_path / "cache_64.pt"
    torch.save(cache, path)
    return str(path)


def _cfg(tiny_cache: str, tmp_path: Path, **kw) -> TrainConfig:
    base = dict(name="t", epochs=2, batch_size=32, cache_path=tiny_cache,
                device="cpu", runs_dir=str(tmp_path / "runs"), sample_every=1,
                warmup_epochs=1)
    base.update(kw)
    return TrainConfig(**base)


def _rows(run_dir: Path) -> list[dict]:
    with open(run_dir / "metrics.csv") as f:
        return list(csv.DictReader(f))


def test_train_writes_expected_artifacts(tiny_cache, tmp_path):
    run_dir = train(_cfg(tiny_cache, tmp_path))
    for name in ("config.json", "metrics.csv", "best.pt", "last.pt", "samples_epoch1.png"):
        assert (run_dir / name).exists(), name
    assert len(_rows(run_dir)) == 4  # 2 epochs x (train, val)
    assert json.loads((run_dir / "config.json").read_text())["beta"] == 1.0


def test_train_refuses_to_overwrite_finished_run(tiny_cache, tmp_path):
    cfg = _cfg(tiny_cache, tmp_path)
    train(cfg)
    with pytest.raises(FileExistsError):
        train(cfg)
    train(cfg, force=True)  # explicit overwrite is allowed


def test_interrupted_run_leaves_usable_checkpoint(tiny_cache, tmp_path):
    def hook(epoch: int) -> None:
        if epoch == 1:
            raise KeyboardInterrupt

    run_dir = train(_cfg(tiny_cache, tmp_path, epochs=5), epoch_hook=hook)
    state = torch.load(run_dir / "last.pt", weights_only=True)
    assert "model" in state
    rows = _rows(run_dir)
    assert {int(r["epoch"]) for r in rows} == {1}
    assert len(rows) == 2


def test_val_logged_at_target_beta_train_at_annealed_beta(tiny_cache, tmp_path):
    run_dir = train(_cfg(tiny_cache, tmp_path, beta=0.5, warmup_epochs=4))
    rows = _rows(run_dir)
    train_betas = [float(r["beta"]) for r in rows if r["split"] == "train"]
    val_betas = [float(r["beta"]) for r in rows if r["split"] == "val"]
    assert train_betas == [0.0, 0.125]  # beta_at_epoch(0..1, 0.5, warmup 4)
    assert val_betas == [0.5, 0.5]


def test_run_finished_is_false_for_interrupted_run_and_true_for_complete_run(tiny_cache, tmp_path):
    def hook(epoch: int) -> None:
        if epoch == 1:
            raise KeyboardInterrupt

    interrupted = train(_cfg(tiny_cache, tmp_path, name="cut", epochs=3), epoch_hook=hook)
    complete = train(_cfg(tiny_cache, tmp_path, name="full", epochs=2))
    assert run_finished(interrupted, epochs=3) is False
    assert run_finished(complete, epochs=2) is True


def test_cli_exits_nonzero_when_training_was_interrupted(tiny_cache, tmp_path, monkeypatch):
    # An interrupted train() returns normally; the CLI must still fail so ./run_all.sh stops.
    def interrupted_train(cfg, force=False, epoch_hook=None):
        return train(cfg, force=force, epoch_hook=lambda e: (_ for _ in ()).throw(KeyboardInterrupt))

    monkeypatch.setattr(train_module, "train", interrupted_train)
    argv = ["--name", "cli", "--epochs", "3", "--batch-size", "32", "--cache-path", tiny_cache,
            "--device", "cpu", "--runs-dir", str(tmp_path / "runs"), "--warmup-epochs", "1"]
    with pytest.raises(SystemExit) as exc:
        train_module.main(argv)
    assert exc.value.code == 130
