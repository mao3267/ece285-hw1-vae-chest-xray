# VAE Chest X-ray Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train convolutional VAEs on the Kaggle chest X-ray dataset, evaluate generated images with FID and IS, and produce a 2-page NeurIPS report with an AI-usage statement.

**Architecture:** A small `vae_xray` package (device utils, data cache, model + loss, training loop, evaluation, plotting). Each stage is a CLI (`python -m vae_xray.<stage>`) that reads and writes files under `runs/<name>/`, so stages can be rerun independently. The report's tables and figures are generated from those files, never typed by hand.

**Tech Stack:** Python 3.12, uv, PyTorch (MPS or CPU), torchvision, pytorch-ignite (FID, InceptionScore), scikit-learn (GaussianMixture, PCA), matplotlib, pandas, pytest, LaTeX (NeurIPS 2025 style, `latexmk`).

**Spec:** `docs/2026-10-23-vae-chest-xray-design.md`

## Global Constraints

- Python `requires-python = ">=3.12,<3.13"`; all commands run through `uv run`.
- Device rule for every script: `mps` if `torch.backends.mps.is_available()`, else `cpu`; no CUDA branch; `--device` overrides.
- Images: grayscale, 64×64 (short side resized to 64, center-crop), float in [0,1]; `--img-size` flag allowed.
- Data: use only `data/chest_xray/{train,val,test}`; ignore `__MACOSX` and the nested duplicate `chest_xray/chest_xray`. Ignore the official 16-image `val`; hold out a seeded 10% of `train` for validation. `test` (624 images) is the FID reference.
- Augmentation (train only): rotation ±5°, small translation, **no horizontal flip**.
- Model: encoder Conv 1→64→128→256→512, kernel 4, stride 2, padding 1, BatchNorm, LeakyReLU(0.2); latent dim 128; decoder mirrored, sigmoid output.
- Loss: BCE summed over pixels, averaged over batch, plus β·KL; β linearly annealed from 0 over the first 10 epochs.
- Training: Adam, lr 1e-3, batch 128, 100 epochs, ReduceLROnPlateau on validation loss. β sweep: {1, 0.1, 0.01}, everything else equal.
- Evaluation: 2,000 prior samples (z ~ N(0, I)) and 2,000 GMM samples (10-component mixture fit on training-set latent means); also FID of test reconstructions; Inception input = 3-channel repeat, bilinear resize to 299×299.
- Report: 2 pages main content, NeurIPS 2025 style, compiles with `latexmk`; includes hyperparameter table, loss curves, FID/IS table, sample images, limitations, and an AI-usage statement naming Claude Code (Claude Opus 5.5 and Claude Sonnet 5.5). Author name left as a TODO.
- Every number in the report comes from `runs/*/eval.json` or `metrics.csv`; fixed seeds; a `config.json` saved per run.
- Commits: one logical change each, staged with explicit paths. The HW1 folder is not a git repo yet, so Task 1 asks the user whether to `git init`; if they decline, skip every commit step.

## Review Focus

Failure modes the spec implies but a person running this is likely to hit, with the task that pins each one:

1. MPS unavailable (or `--device cpu`): scripts must still run end to end on CPU. (Task 1)
2. The Kaggle zip contains junk: `__MACOSX/`, `.DS_Store`, a nested duplicate `chest_xray/` folder, and some RGB-mode JPEGs among grayscale ones. Counts must come out to 5,216 train and 624 test, all single-channel. (Task 2)
3. KL collapse or NaN: with β = 1 the posterior can collapse to the prior, and `logvar` can overflow. Loss must stay finite on extreme inputs, and KL must be nonzero after a short training run. (Task 3)
4. Interrupted training: a run killed mid-way must leave a usable `last.pt` and a `metrics.csv` with one row per finished epoch; rerunning with the same `--name` must not silently overwrite finished runs. (Task 4)
5. FID/IS on MPS: ignite's FID accumulates in float64, which MPS does not support, and InceptionV3 may hit MPS gaps. Features are computed on the selected device, then metric state is kept on CPU. (Task 5)

## File Structure

```
pyproject.toml  README.md  run_all.sh
src/vae_xray/__init__.py
src/vae_xray/common.py      # get_device, seed_everything
src/vae_xray/data.py        # zip -> cache, splits, loaders
src/vae_xray/model.py       # VAE, vae_loss, beta_at_epoch
src/vae_xray/train.py       # TrainConfig, train(), CLI
src/vae_xray/evaluate.py    # Inception features, FID/IS, GMM sampler, CLI
src/vae_xray/plots.py       # figures + results table (.tex), CLI
tests/test_common.py  tests/test_data.py  tests/test_model.py
tests/test_train.py   tests/test_evaluate.py
report/main.tex  report/references.bib  report/neurips_2025.sty  report/figures/
```

---

### Task 1: Project scaffold and device utilities

**Files:**
- Create: `pyproject.toml`, `src/vae_xray/__init__.py`, `src/vae_xray/common.py`, `tests/test_common.py`

**Interfaces:**
- Produces: `get_device(override: str | None = None) -> torch.device`; `seed_everything(seed: int) -> None`

- [ ] **Step 1:** Ask the user whether to run `git init` in `HW1/` (and add a `.gitignore` for `data/`, `runs/`, `.venv/`, `*.zip`, `__pycache__/`). Record the answer; it decides whether later commit steps run.
- [ ] **Step 2:** `uv init --lib` style project pinned to Python 3.12 with the dependencies in the Tech Stack line (pytest as a dev dependency); package under `src/vae_xray`. Run `uv sync` and confirm `uv run python -c "import torch, ignite, sklearn; print(torch.__version__)"` prints a version.
- [ ] **Step 3: Write the failing tests** in `tests/test_common.py`:
  - `test_get_device_prefers_mps_when_available`: monkeypatch `torch.backends.mps.is_available` to `True` → `get_device().type == "mps"`.
  - `test_get_device_falls_back_to_cpu`: monkeypatch to `False` → `get_device().type == "cpu"`.
  - `test_get_device_override`: `get_device("cpu").type == "cpu"` even when MPS is "available".
  - `test_seed_everything_reproducible`: seed twice → identical `torch.rand(3)`.
- [ ] **Step 4:** Run `uv run pytest tests/test_common.py -v`; expect FAIL (module missing).
- [ ] **Step 5:** Implement `get_device` and `seed_everything` (seeds `random`, `numpy`, `torch`) in `common.py`.
- [ ] **Step 6:** Run `uv run pytest tests/test_common.py -v`; expect 4 passed.
- [ ] **Step 7:** (If git) commit `pyproject.toml`, `.gitignore`, `src/vae_xray/__init__.py`, `src/vae_xray/common.py`, `tests/test_common.py`.

### Task 2: Data cache and loaders

**Files:**
- Create: `src/vae_xray/data.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: `seed_everything` from Task 1.
- Produces:
  - `prepare_cache(zip_path: Path, data_dir: Path, img_size: int = 64, seed: int = 0, val_frac: float = 0.1) -> Path` — extracts the zip into `data_dir` (skipping `__MACOSX` and non-image files), builds `data_dir/cache_{img_size}.pt`, returns that path. Skips work if the cache already exists.
  - `load_cache(path: Path) -> dict[str, torch.Tensor]` with keys `train_x`, `val_x`, `test_x` (uint8, shape `N×1×H×W`) and `train_y`, `val_y`, `test_y` (int64, 0 = NORMAL, 1 = PNEUMONIA).
  - `make_loaders(cache: dict, batch_size: int = 128, augment: bool = True, num_workers: int = 0) -> tuple[DataLoader, DataLoader]` returning (train, val); batches are float32 in [0,1], shape `B×1×H×W`, yielding `(x, y)`.

- [ ] **Step 1: Write failing tests** using a tiny synthetic zip built in `tmp_path` (a few JPEGs under `chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}`, one RGB-mode JPEG, plus `__MACOSX/` junk and a `.DS_Store`):
  - `test_prepare_cache_ignores_junk_and_is_single_channel`: cache tensors have 1 channel; counts equal the number of real images only.
  - `test_val_split_is_seeded_and_disjoint`: two calls with the same seed give the same split; train/val sizes are 90% / 10% of train (rounded); no image appears in both.
  - `test_official_val_folder_is_ignored`: images from `val/` appear in none of the three splits.
  - `test_loader_batches_are_float_in_unit_range`: first batch is `float32`, min ≥ 0, max ≤ 1, shape `(B, 1, 64, 64)`.
  - `test_augmentation_only_in_train_loader`: with `augment=True`, two passes over the val loader are identical.
- [ ] **Step 2:** Run `uv run pytest tests/test_data.py -v`; expect FAIL.
- [ ] **Step 3:** Implement in `data.py`. Use `PIL.Image.open(...).convert("L")`, resize the short side to `img_size` then center-crop, store as uint8. Augmentation: `torchvision.transforms.v2.RandomAffine(degrees=5, translate=(0.05, 0.05))` applied per sample in the train dataset's `__getitem__`; no flip.
- [ ] **Step 4:** Run the tests; expect all pass.
- [ ] **Step 5:** Once the Kaggle zip finishes downloading and `unzip -tq` reports no errors, run `prepare_cache` on `~/Downloads/chest-xray-pneumonia.zip` into `data/`, and verify with a short script: train+val total = 5,216, test = 624, all tensors have 1 channel. Expected output: `train 4694 val 522 test 624` (rounding of the 10% split may shift these by one).
- [ ] **Step 6:** (If git) commit `src/vae_xray/data.py`, `tests/test_data.py`.

### Task 3: VAE model and loss

**Files:**
- Create: `src/vae_xray/model.py`, `tests/test_model.py`

**Interfaces:**
- Produces:
  - `class VAE(nn.Module)`: `__init__(self, latent_dim: int = 128, widths: tuple[int, ...] = (64, 128, 256, 512), img_size: int = 64)`; `encode(x) -> tuple[Tensor, Tensor]` (`mu`, `logvar`); `reparameterize(mu, logvar) -> Tensor`; `decode(z) -> Tensor` (sigmoid, in [0,1]); `forward(x) -> tuple[Tensor, Tensor, Tensor]` (`x_hat`, `mu`, `logvar`); `sample(n: int, device) -> Tensor`.
  - `vae_loss(x, x_hat, mu, logvar, beta: float) -> dict[str, Tensor]` with scalar tensors `total`, `recon`, `kl`, each averaged over the batch (recon and kl summed over pixels / latent dims per sample).
  - `beta_at_epoch(epoch: int, target: float, warmup_epochs: int = 10) -> float`: `target * min(1, epoch / warmup_epochs)` (epoch is 0-indexed).

- [ ] **Step 1: Write failing tests:**
  - `test_forward_shapes`: input `(4,1,64,64)` → `x_hat (4,1,64,64)`, `mu` and `logvar` `(4,128)`.
  - `test_decode_output_in_unit_range`.
  - `test_kl_is_zero_for_standard_normal_posterior`: `mu=0, logvar=0` → `kl ≈ 0` (atol 1e-6).
  - `test_kl_matches_closed_form`: random `mu, logvar` vs `-0.5 * sum(1 + logvar - mu² - exp(logvar))` averaged over batch.
  - `test_total_equals_recon_plus_beta_kl`.
  - `test_loss_finite_for_extreme_logvar`: `logvar = ±50` still yields finite loss (the model clamps `logvar` to [−10, 10] in `encode`).
  - `test_beta_schedule`: `beta_at_epoch(0, 1.0) == 0`, `beta_at_epoch(5, 1.0) == 0.5`, `beta_at_epoch(10, 1.0) == 1.0`, `beta_at_epoch(50, 0.1) == 0.1`.
  - `test_overfit_smoke`: 64 random-ish images, 150 Adam steps (lr 1e-3, β = 1) → final reconstruction loss at least 30% below the first step's, and mean KL > 1 (posterior did not collapse).
- [ ] **Step 2:** Run `uv run pytest tests/test_model.py -v`; expect FAIL.
- [ ] **Step 3:** Implement `model.py` per the signatures. Use `F.binary_cross_entropy(x_hat, x, reduction="sum") / B` for `recon`. The encoder flattens `4×4×512`; two `nn.Linear` heads; the decoder starts with `nn.Linear(latent_dim, 4*4*512)`.
- [ ] **Step 4:** Run the tests; expect all pass.
- [ ] **Step 5:** (If git) commit `src/vae_xray/model.py`, `tests/test_model.py`.

### Task 4: Training loop

**Files:**
- Create: `src/vae_xray/train.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: `get_device`, `seed_everything`; `load_cache`, `make_loaders`; `VAE`, `vae_loss`, `beta_at_epoch`.
- Produces:
  - `@dataclass TrainConfig`: `name: str`, `beta: float = 1.0`, `epochs: int = 100`, `batch_size: int = 128`, `lr: float = 1e-3`, `latent_dim: int = 128`, `img_size: int = 64`, `warmup_epochs: int = 10`, `seed: int = 0`, `cache_path: str = "data/cache_64.pt"`, `device: str | None = None`, `runs_dir: str = "runs"`, `sample_every: int = 10`.
  - `train(cfg: TrainConfig) -> Path` returns the run directory `runs/<name>/` containing `config.json`, `metrics.csv` (columns `epoch, split, total, recon, kl, beta, lr`; one `train` and one `val` row per epoch), `best.pt`, `last.pt`, and `samples_epoch<N>.png` every `sample_every` epochs using a fixed noise tensor.
  - CLI: `uv run python -m vae_xray.train --name beta1 --beta 1.0 [--epochs N ...]`.

- [ ] **Step 1: Write failing tests** with a tiny cache built in memory (e.g. 64 train / 16 val random images saved to `tmp_path`):
  - `test_train_writes_expected_artifacts`: 2 epochs on CPU → run dir has `config.json`, `metrics.csv` (4 rows), `best.pt`, `last.pt`.
  - `test_train_refuses_to_overwrite_finished_run`: a second `train(cfg)` call with the same `name` raises `FileExistsError`; `train(cfg, force=True)` (CLI `--force`) overwrites.
  - `test_interrupted_run_leaves_usable_checkpoint`: run with an epoch hook that raises `KeyboardInterrupt` after epoch 1 → `last.pt` loads, and `metrics.csv` has exactly the finished epochs' rows.
  - `test_val_loss_logged_without_augmentation_and_with_full_beta`: validation rows record the loss at the *target* β, so val curves are comparable across epochs (train rows record the annealed β).
- [ ] **Step 2:** Run `uv run pytest tests/test_train.py -v`; expect FAIL.
- [ ] **Step 3:** Implement `train()`: per-epoch train pass (annealed β), eval pass (target β, `torch.no_grad`), `ReduceLROnPlateau(patience=5, factor=0.5)` on val total, write the CSV row after each epoch (flush), save `last.pt` every epoch and `best.pt` on a new best val total. Catch `KeyboardInterrupt` after saving.
- [ ] **Step 4:** Run the tests; expect all pass.
- [ ] **Step 5:** Time a real run: `uv run python -m vae_xray.train --name timing --epochs 2`; record seconds per epoch and the device used; delete `runs/timing`. Expected: device `mps` on the M1; if slower than ~60 s/epoch, tell the user before launching the sweep.
- [ ] **Step 6:** (If git) commit `src/vae_xray/train.py`, `tests/test_train.py`.

### Task 5: Evaluation (FID, IS, GMM sampler)

**Files:**
- Create: `src/vae_xray/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `VAE`, `get_device`, `load_cache`.
- Produces:
  - `to_inception_input(x: Tensor) -> Tensor`: `B×1×H×W` in [0,1] → `B×3×299×299`, ImageNet-normalized as ignite's example does.
  - `inception_outputs(x: Tensor, device, batch_size: int = 64) -> tuple[Tensor, Tensor]`: pooled features (`N×2048`) and class logits (`N×1000`) on CPU, from torchvision InceptionV3 with ImageNet weights, run on `device`.
  - `fid_and_is(real: Tensor, fake: Tensor, device) -> dict[str, float]` with keys `fid`, `is_mean`, `is_std`, using `ignite.metrics.FID` and `ignite.metrics.InceptionScore` with metric state on CPU.
  - `fit_latent_gmm(mu: np.ndarray, k: int = 10, seed: int = 0) -> GaussianMixture` and `sample_gmm(gmm, n: int) -> Tensor`.
  - `evaluate_run(run_dir: Path, n_samples: int = 2000, device: str | None = None) -> dict` loads `best.pt`, computes metrics for `prior`, `gmm`, and `recon` (test reconstructions) against the test set, writes `eval.json`, and returns it. Keys: `prior`, `gmm`, `recon`, each with `fid`, `is_mean`, `is_std`; plus `n_samples`, `n_real`, `device`.
  - CLI: `uv run python -m vae_xray.evaluate --run beta1`.

- [ ] **Step 1: Write failing tests** (use CPU and tiny inputs; stub the Inception network with a small fake extractor via a monkeypatchable `load_inception()` so tests are fast and offline):
  - `test_to_inception_input_shape_and_channels`: `(2,1,64,64)` → `(2,3,299,299)`.
  - `test_fid_of_identical_sets_is_near_zero`: same tensor as real and fake → `fid < 1e-3` (stubbed extractor).
  - `test_fid_grows_with_distribution_shift`: noise vs shifted noise has larger FID than noise vs noise.
  - `test_fit_and_sample_gmm_shapes`: `mu (500,16)`, `k=3` → `sample_gmm(gmm, 40)` has shape `(40,16)`.
  - `test_metric_state_stays_on_cpu`: after `fid_and_is` with `device="cpu"` result values are Python floats (guards the float64/MPS issue).
- [ ] **Step 2:** Run `uv run pytest tests/test_evaluate.py -v`; expect FAIL.
- [ ] **Step 3:** Implement `evaluate.py`. Follow the pytorch-ignite blog's FID/IS usage (features from InceptionV3 fed into the metrics' `update`). If an op fails on MPS, set `PYTORCH_ENABLE_MPS_FALLBACK=1` inside the CLI entry point and print a note naming the failing op. The GMM is fit on the encoder means of the training-set images (eval mode, no augmentation).
- [ ] **Step 4:** Run the tests; expect all pass.
- [ ] **Step 5:** Smoke-run against a real checkpoint once one exists (after Task 7 Step 1): `uv run python -m vae_xray.evaluate --run beta1 --n-samples 200`. Expected: writes `runs/beta1/eval.json` with finite numbers and a `device` field.
- [ ] **Step 6:** (If git) commit `src/vae_xray/evaluate.py`, `tests/test_evaluate.py`.

### Task 6: Figures and results table

**Files:**
- Create: `src/vae_xray/plots.py`, `tests/test_plots.py`

**Interfaces:**
- Consumes: run directories containing `metrics.csv`, `best.pt`, `eval.json`; `VAE`; `load_cache`.
- Produces (all under an `out_dir`, default `report/figures/`):
  - `plot_losses(run_dirs: list[Path], out: Path) -> None` — train vs val total/recon/KL curves per run.
  - `plot_samples(run_dir: Path, out: Path, source: str = "prior") -> None` — 8×8 grid, `source` ∈ {`prior`, `gmm`}.
  - `plot_recon(run_dir: Path, out: Path, n: int = 8) -> None` — real row above reconstruction row (test images).
  - `plot_interp(run_dir: Path, out: Path, steps: int = 10) -> None` — latent interpolation strip between two test images.
  - `plot_pca(run_dir: Path, out: Path) -> None` — PCA of test latent means colored by class (optional figure).
  - `write_results_table(run_dirs: list[Path], out: Path) -> None` — LaTeX `tabular` with one row per run and columns β, prior FID, GMM FID, recon FID, IS (mean ± std); values read from each `eval.json`.
  - CLI: `uv run python -m vae_xray.plots --runs beta1 beta0.1 beta0.01 --best beta0.1`.

- [ ] **Step 1: Write failing tests** with a fake run directory (tiny `metrics.csv` and `eval.json` fixtures in `tmp_path`):
  - `test_write_results_table_contains_values_from_eval_json`: generated `.tex` contains each run's FID formatted to one decimal and has one row per run.
  - `test_plot_losses_writes_nonempty_file`.
  - `test_write_results_table_fails_loudly_on_missing_eval`: missing `eval.json` raises `FileNotFoundError` (no silent blanks in the report).
- [ ] **Step 2:** Run `uv run pytest tests/test_plots.py -v`; expect FAIL.
- [ ] **Step 3:** Implement `plots.py` with matplotlib (Agg backend), PDF output for line plots and PNG for image grids (`torchvision.utils.make_grid`). Use one consistent grayscale colormap for image panels.
- [ ] **Step 4:** Run the tests; expect all pass.
- [ ] **Step 5:** (If git) commit `src/vae_xray/plots.py`, `tests/test_plots.py`.

### Task 7: Experiments (β sweep, evaluation, figures)

**Files:**
- Create: `run_all.sh`
- Produces: `runs/beta1/`, `runs/beta0.1/`, `runs/beta0.01/` (with `eval.json`), `report/figures/*`, `report/results_table.tex`.

- [ ] **Step 1:** Run β = 1 first: `uv run python -m vae_xray.train --name beta1 --beta 1.0`. Check that the validation loss decreases, the KL is clearly above 0, and `samples_epoch*.png` look like chest X-rays (not noise). If the KL collapses to ~0, stop and tell the user before continuing.
- [ ] **Step 2:** Run `uv run python -m vae_xray.evaluate --run beta1`.
- [ ] **Step 3:** Write `run_all.sh` that, for β in 1, 0.1, 0.01, trains `beta<β>` (skipping runs that already finished) and then evaluates each; run it for the remaining two β values. Run it in the background and report timing to the user.
- [ ] **Step 4:** Pick the best run by prior or GMM FID (report both; say which one is the headline). Run `uv run python -m vae_xray.plots --runs beta1 beta0.1 beta0.01 --best <name>`. Check `report/figures/` and `report/results_table.tex` exist.
- [ ] **Step 5:** Open the figures and inspect them (loss curves converge, sample grid plausible). Note anything surprising for the report's discussion, and only describe class separation in the PCA plot if it is actually visible.
- [ ] **Step 6:** (If git) commit `run_all.sh`.

### Task 8: Report (NeurIPS format, 2 pages)

**Files:**
- Create: `report/main.tex`, `report/references.bib`; download `report/neurips_2025.sty` from the official NeurIPS site.
- Test: compile with `latexmk`.

**Interfaces:**
- Consumes: `report/figures/*`, `report/results_table.tex` from Task 7; the Chandra et al. 2025 paper for related work.

- [ ] **Step 1:** Fetch the NeurIPS 2025 style file into `report/` (ask the user first, since it is a download; state the source and file name). Use the `preprint`-style option unless the user wants `final`.
- [ ] **Step 2:** Write `main.tex` with sections: Introduction (with the prior-work paragraph), Method (VAE, loss, β annealing, GMM sampler), Experimental setup (hyperparameter table), Results (loss-curve figure, `\input{results_table.tex}`, sample grid and reconstruction figure), Discussion and limitations (blurriness, FID on 624 real images biased upward, IS limited for X-rays, any claims limited to what the plots show), **AI usage statement** (Claude Code with Opus 5.5 and Sonnet 5.5 wrote code and drafted the report and helped find and summarize related work; the student directed the work, ran the experiments, checked results, edited the text), References. Leave `\author{TODO: your name}`.
- [ ] **Step 3:** Fill `references.bib`: Kingma & Welling 2014, Kermany et al. 2018, Heusel et al. 2017, Salimans et al. 2016, Ghosh et al. 2020, Higgins et al. 2017, Chandra et al. 2025 (DOI 10.1109/CENIM67940.2025.11326136), pytorch-ignite. Verify each entry's title, authors and year against its source before keeping it.
- [ ] **Step 4:** Compile: `cd report && latexmk -pdf main.tex`. Expected: `main.pdf` builds with no undefined references; main content ≤ 2 pages (references and the AI statement may spill onto page 3 only if the user's course allows it; otherwise trim figures). Check page count with `pdfinfo main.pdf`.
- [ ] **Step 5:** Cross-check every number and claim in the text against `eval.json` and the figures; fix mismatches.
- [ ] **Step 6:** (If git) commit `report/main.tex`, `report/references.bib`, `report/neurips_2025.sty`, `report/figures`, `report/results_table.tex`.

### Task 9: README and final verification

**Files:**
- Create: `README.md`

- [ ] **Step 1:** Write the README: setup (`uv sync`), data location and unzip step, the exact commands for `train`, `evaluate`, `plots`, `latexmk`, and how to pick the device.
- [ ] **Step 2:** Run `uv run pytest -v`; expect every test to pass.
- [ ] **Step 3:** Run `uvx ruff check src tests`; fix any `F`/`E9` findings.
- [ ] **Step 4:** Verify from a clean shell that the README commands work, using a 2-epoch run into a scratch run name; delete the scratch run.
- [ ] **Step 5:** Build the Gradescope bundle list for the user (code: `src/`, `tests/`, `pyproject.toml`, `README.md`, `run_all.sh`; report: `report/main.pdf`) without zipping `data/` or `runs/` checkpoints. Remind the user of the TODOs: author name, style option, the AI-policy wording check.
- [ ] **Step 6:** (If git) commit `README.md`.

---

## Self-review notes

- **Spec coverage:** device rule (Task 1), data rules (2), model/loss/β schedule (3), training and artifacts (4), prior/GMM/recon FID and IS (5), figures and table (6), β sweep (7), report with AI statement (8), README (9).
- **Open assumption:** the exact ignite call pattern for feeding precomputed features is confirmed at implementation time (Task 5 Step 3); if ignite can't accept precomputed features cleanly, the fallback is to pass raw images through ignite's own extractor on CPU, which is slower but unchanged in output.
