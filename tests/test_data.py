import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vae_xray.data import load_cache, make_loaders, prepare_cache

N_TRAIN_PER_CLASS = 20
N_TEST_PER_CLASS = 3
N_OFFICIAL_VAL_PER_CLASS = 2


def _write_jpeg(zf: zipfile.ZipFile, name: str, mode: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    shape = (90, 120) if mode == "L" else (90, 120, 3)
    img = Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8), mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    zf.writestr(name, buf.getvalue())


@pytest.fixture
def fake_zip(tmp_path: Path) -> Path:
    zip_path = tmp_path / "fake.zip"
    seed = 0
    with zipfile.ZipFile(zip_path, "w") as zf:
        for split, n in [
            ("train", N_TRAIN_PER_CLASS),
            ("test", N_TEST_PER_CLASS),
            ("val", N_OFFICIAL_VAL_PER_CLASS),
        ]:
            for cls in ("NORMAL", "PNEUMONIA"):
                for i in range(n):
                    seed += 1
                    # one RGB-mode JPEG among the grayscale ones
                    mode = "RGB" if (split == "train" and cls == "NORMAL" and i == 0) else "L"
                    _write_jpeg(zf, f"chest_xray/{split}/{cls}/img_{i}.jpeg", mode, seed)
        # junk that must be ignored
        _write_jpeg(zf, "__MACOSX/chest_xray/train/NORMAL/._img_0.jpeg", "L", 999)
        _write_jpeg(zf, "chest_xray/__MACOSX/chest_xray/train/NORMAL/._img_0.jpeg", "L", 997)
        _write_jpeg(zf, "chest_xray/chest_xray/train/NORMAL/dup_0.jpeg", "L", 998)
        zf.writestr("chest_xray/train/NORMAL/.DS_Store", b"junk")
    return zip_path


def test_prepare_cache_ignores_junk_and_is_single_channel(fake_zip, tmp_path):
    cache = load_cache(prepare_cache(fake_zip, tmp_path / "data", img_size=64, seed=0))
    n_train_total = 2 * N_TRAIN_PER_CLASS
    assert len(cache["train_x"]) + len(cache["val_x"]) == n_train_total
    assert len(cache["test_x"]) == 2 * N_TEST_PER_CLASS
    for key in ("train_x", "val_x", "test_x"):
        assert cache[key].dtype == torch.uint8
        assert cache[key].shape[1:] == (1, 64, 64)


def test_extraction_skips_nested_junk_and_duplicate_folder(fake_zip, tmp_path):
    data_dir = tmp_path / "data"
    prepare_cache(fake_zip, data_dir, seed=0)
    assert not list(data_dir.rglob("__MACOSX"))
    assert not (data_dir / "chest_xray" / "chest_xray").exists()


def test_val_split_is_seeded_and_disjoint(fake_zip, tmp_path):
    a = load_cache(prepare_cache(fake_zip, tmp_path / "a", seed=0))
    b = load_cache(prepare_cache(fake_zip, tmp_path / "b", seed=0))
    assert torch.equal(a["val_x"], b["val_x"])
    assert len(a["val_x"]) == round(0.1 * 2 * N_TRAIN_PER_CLASS)
    train = {x.numpy().tobytes() for x in a["train_x"]}
    val = {x.numpy().tobytes() for x in a["val_x"]}
    assert not train & val


def test_official_val_folder_is_ignored(fake_zip, tmp_path):
    cache = load_cache(prepare_cache(fake_zip, tmp_path / "data", seed=0))
    total = len(cache["train_x"]) + len(cache["val_x"]) + len(cache["test_x"])
    assert total == 2 * N_TRAIN_PER_CLASS + 2 * N_TEST_PER_CLASS


def test_loader_batches_are_float_in_unit_range(fake_zip, tmp_path):
    cache = load_cache(prepare_cache(fake_zip, tmp_path / "data", seed=0))
    train_loader, _ = make_loaders(cache, batch_size=8, augment=True)
    x, y = next(iter(train_loader))
    assert x.dtype == torch.float32
    assert x.min() >= 0 and x.max() <= 1
    assert x.shape == (8, 1, 64, 64)
    assert y.shape == (8,)


def test_augmentation_only_in_train_loader(fake_zip, tmp_path):
    cache = load_cache(prepare_cache(fake_zip, tmp_path / "data", seed=0))
    _, val_loader = make_loaders(cache, batch_size=4, augment=True)
    first = torch.cat([x for x, _ in val_loader])
    second = torch.cat([x for x, _ in val_loader])
    assert torch.equal(first, second)
