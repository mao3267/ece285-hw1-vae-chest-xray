# ECE 285 HW1: Convolutional VAE for chest X-ray generation

Trains convolutional VAEs on the Kaggle
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
dataset, then scores generated images with FID and Inception Score (PyTorch-Ignite).
Design: `docs/2026-10-23-vae-chest-xray-design.md`. Report: `report/main.pdf`.

## Setup

```bash
uv sync                      # Python 3.12 + dependencies
```

Every script picks its device automatically: `mps` if available, otherwise `cpu`
(override with `--device cpu`). On Apple GPUs a 100-epoch run takes about 14 minutes.

## Data

Download the Kaggle zip (about 2.3 GB) and build the 64x64 grayscale cache once:

```bash
uv run python -c "from pathlib import Path; from vae_xray.data import prepare_cache; \
prepare_cache(Path('~/Downloads/chest-xray-pneumonia.zip').expanduser(), Path('data'))"
```

This extracts only `chest_xray/{train,test}` into `data/` and writes `data/cache_64.pt`
(4,694 train / 522 validation images held out from `train`, plus the 624 test images).

## Run

```bash
./run_all.sh                 # trains beta = 1, 0.1, 0.01 without augmentation (runs/beta*_noaug) and evaluates each
./run_all.sh train           # training only
./run_all.sh eval            # evaluation only

# a single run (training augmentation is on by default; the report's sweep uses --no-augment)
uv run python -m vae_xray.train --name beta1_noaug --beta 1.0 --no-augment
uv run python -m vae_xray.evaluate --run beta1_noaug

# the augmentation ablation reported in the paper
uv run python -m vae_xray.train --name beta0.01 --beta 0.01
uv run python -m vae_xray.evaluate --run beta0.01
```

Finished runs are skipped, so `./run_all.sh` is safe to re-run after an interruption.

Each run writes `runs/<name>/` with `config.json`, `metrics.csv`, `best.pt`, `last.pt`,
sample grids, and (after evaluation) `eval.json`. Evaluation downloads the ImageNet
InceptionV3 weights (about 104 MB) on first use.

## Figures, table and report

```bash
uv run python -m vae_xray.plots --runs beta1_noaug beta0.1_noaug beta0.01_noaug --best beta0.01_noaug
cd report && latexmk -pdf main.tex
```

The figures in `report/figures/` and `report/results_table.tex` are generated from the
run files, so every number in the report comes from `eval.json` / `metrics.csv`.

## Tests

```bash
uv run pytest -q
```

## Layout

```
src/vae_xray/   common.py (device, seed)  data.py  model.py  train.py  evaluate.py  plots.py
tests/          unit and smoke tests (FID/IS tests use an offline stand-in for InceptionV3)
report/         main.tex, references.bib, neurips_2025.sty, figures/, main.pdf
docs/           design and implementation plan
```
