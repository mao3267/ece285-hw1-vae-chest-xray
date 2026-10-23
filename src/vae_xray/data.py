"""Dataset preparation: Kaggle zip -> cached uint8 tensors, plus loaders."""

import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from vae_xray.common import seed_everything

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CLASSES = ("NORMAL", "PNEUMONIA")


def _extract(zip_path: Path, data_dir: Path) -> None:
    """Extract only chest_xray/{train,test}/..., skipping __MACOSX junk and the nested duplicate."""
    keep = ("chest_xray/train/", "chest_xray/test/")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist()
                   if m.startswith(keep) and "__MACOSX" not in m]
        zf.extractall(data_dir, members=members)


def _list_split(root: Path, split: str) -> list[tuple[Path, int]]:
    """Image paths and labels under root/chest_xray/<split>/<CLASS>/ only."""
    items = []
    for label, cls in enumerate(CLASSES):
        folder = root / "chest_xray" / split / cls
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("."):
                items.append((p, label))
    return items


def _load_image(path: Path, img_size: int) -> np.ndarray:
    """Grayscale, resize the short side to img_size, center-crop to a square."""
    img = Image.open(path).convert("L")
    w, h = img.size
    scale = img_size / min(w, h)
    img = img.resize((max(img_size, round(w * scale)), max(img_size, round(h * scale))),
                     Image.BILINEAR)
    w, h = img.size
    left, top = (w - img_size) // 2, (h - img_size) // 2
    return np.asarray(img.crop((left, top, left + img_size, top + img_size)), dtype=np.uint8)


def _to_tensors(items: list[tuple[Path, int]], img_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.stack([_load_image(p, img_size) for p, _ in items])[:, None]
    y = np.array([label for _, label in items], dtype=np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def prepare_cache(zip_path: Path, data_dir: Path, img_size: int = 64, seed: int = 0,
                  val_frac: float = 0.1) -> Path:
    """Extract the zip and build data_dir/cache_<img_size>.pt (skipped if it exists)."""
    zip_path, data_dir = Path(zip_path), Path(data_dir)
    cache_path = data_dir / f"cache_{img_size}.pt"
    if cache_path.exists():
        return cache_path
    data_dir.mkdir(parents=True, exist_ok=True)
    _extract(zip_path, data_dir)

    train_x, train_y = _to_tensors(_list_split(data_dir, "train"), img_size)
    test_x, test_y = _to_tensors(_list_split(data_dir, "test"), img_size)

    seed_everything(seed)
    perm = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(seed))
    n_val = round(val_frac * len(train_x))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    cache = {
        "train_x": train_x[train_idx], "train_y": train_y[train_idx],
        "val_x": train_x[val_idx], "val_y": train_y[val_idx],
        "test_x": test_x, "test_y": test_y,
    }
    torch.save(cache, cache_path)
    return cache_path


def load_cache(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, weights_only=True)


class _XrayDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, augment: bool):
        self.x, self.y = x, y
        # Rotation and translation only; no flip (it would mirror the anatomy).
        self.aug = v2.RandomAffine(degrees=5, translate=(0.05, 0.05)) if augment else None

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[i]
        if self.aug is not None:
            x = self.aug(x)
        return x.float() / 255.0, self.y[i]


def make_loaders(cache: dict, batch_size: int = 128, augment: bool = True,
                 num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    """Return (train, val) loaders; augmentation applies to train only."""
    train = _XrayDataset(cache["train_x"], cache["train_y"], augment)
    val = _XrayDataset(cache["val_x"], cache["val_y"], augment=False)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
