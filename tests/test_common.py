import torch

from vae_xray.common import get_device, seed_everything


def test_get_device_prefers_mps_when_available(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_device().type == "mps"


def test_get_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert get_device().type == "cpu"


def test_get_device_override(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_device("cpu").type == "cpu"


def test_seed_everything_reproducible():
    seed_everything(123)
    a = torch.rand(3)
    seed_everything(123)
    b = torch.rand(3)
    assert torch.equal(a, b)
