# Change Log

## 1. Local HTTP Server in Pipeline
- **File:** `run_pipeline.sh`
- Added Step 4 that starts `python -m http.server 8000` after data export, fixing CORS errors when opening `index.html` directly via `file://`.

## 2. Zoom on Latent Space Plot
- **File:** `index.html`
- Added D3 zoom behavior (scroll to zoom, drag to pan) on the latent space scatter plot.
- Added a clip path so dots stay within the plot area when zoomed.
- Saved axis group references (`xAxisG`, `yAxisG`) and wrapped dots in a clipped `<g>` for zoom redraws.

## 3. More Training Data
- **File:** `solve_lotka_volterra.py`
- Changed from 4 hardcoded alpha values to 20 evenly spaced values via `np.linspace(0.1, 0.8, 20)`, giving 1,000 training samples instead of 200.

## 4. Beta Annealing
- **File:** `vae.py` — Added `beta_anneal_epochs` parameter to `VAE.train_model`. Beta linearly ramps from 0 to target over the specified number of epochs.
- **File:** `train_vae_lotkavolt.py` — Set `beta_anneal_epochs=100` (ramp over first 100 of 500 epochs).

## 5. Filtered Display Alphas
- **File:** `export_for_d3.py`
- Training uses 20 alphas, but visualization only shows samples from the 4 closest to the original values (0.1, 0.4, 0.6, 0.8). Keeps the scatter plot clean while benefiting from denser training coverage.

## 6. Dense Interpolation Grid
- **File:** `export_for_d3.py`
- Increased grid from 20x20 to 100x100 (10,000 decode points) so clicking anywhere in the latent space snaps to a very nearby grid point.

## 7. Neural ODE Decoder — Attempted and Reverted
- The `NeuralODEVAE` class was implemented in `vae.py` and then removed; training/export scripts were restored to use the plain `VAE`. The approach remains documented in `improvements.md` as a future option.

## 8. Deeper Network
- **File:** `vae.py`
- Added an extra 256-unit layer at both ends of the VAE, so the architecture is now `700 -> 256 -> 128 -> 64 -> 2 latent -> 64 -> 128 -> 256 -> 700` (previously `700 -> 128 -> 64 -> 2 -> 64 -> 128 -> 700`). Each new linear layer is followed by `BatchNorm1d` + `Tanh`, matching the existing blocks.

## 9. Client-Side Decoding via ONNX Runtime Web
- **File:** `export_for_d3.py` — Exports `model.decoder` to `decoder.onnx` (opset 14, dynamic batch axis) and now includes `x_mean`/`x_std` in `data_for_d3.json` for denormalization. Removed the 100x100 precomputed decode grid, shrinking the JSON substantially.
- **File:** `index.html` — Loads `onnxruntime-web` from CDN and creates an `InferenceSession` alongside the JSON fetch. Empty-space clicks now run the decoder live on the clicked (z1, z2) instead of snapping to the nearest grid point; the 700-dim output is denormalized in JS and split into prey/predator series.
- **Verified:** PyTorch decode vs ONNX decode match to ~1e-6 on the test points checked.

## 10. Sequential Window Sampling
- Changed data generation s.t. instead of sampling random windows per trajectory we sample sequential windows which results in more uniform points in the latent space.

## 11. FitzHugh-Nagumo Support — Dual-Model Visualization
- **File:** `solve_fitzhughnagumo.py` — Fixed missing `torch.save` call; the script now writes `fitzhughnagumo_data.pt` (previously it only plotted and exited). Uses 20 linearly spaced `a` values over [0.10, 0.70] and the same sequential windowing scheme as the Lotka-Volterra pipeline.
- **File:** `train_vae_fitzhughnagumo.py` *(new)* — Mirrors `train_vae_lotkavolt.py` for the FHN dataset; loads `fitzhughnagumo_data.pt`, trains a 2-D latent VAE with `beta_anneal_epochs=100`, and saves weights to `vae_fitzhughnagumo.pt`.
- **File:** `export_for_d3_fitzhughnagumo.py` *(new)* — Same export logic as `export_for_d3.py`; outputs `fhn_data_for_d3.json` and `fhn_decoder.onnx`. Display alphas are `[0.1, 0.3, 0.5, 0.7]`. Reuses `prey`/`predator` JSON keys so the frontend needs no schema changes.
- **File:** `index.html` — Refactored into a generic dual-model explorer. A switcher in the header toggles between Lotka-Volterra and FitzHugh-Nagumo; each model lazy-loads its JSON + ONNX session on first use and caches it for instant re-switching. Accent color, axis labels (*Prey/Predator* vs *Voltage/Recovery*), and crosshair color all update per model. Double-click to reset zoom restored.
- **File:** `run_pipeline.sh` — Extended to run both ODE solvers, both training scripts, and both export scripts before starting the HTTP server.

## 12. Script Consolidation — Unified Pipeline Scripts
- Reduced the number of per-model scripts from 6 down to 3 by merging duplicate files into single scripts driven by a `--model` CLI flag. `vae.py` and `index.html` are unchanged.
- **File:** `solve_ode.py` *(replaces `solve_lotka_volterra.py` and `solve_fitzhughnagumo.py`)* — Both ODE right-hand sides, initial conditions, alpha ranges, and output filenames are declared in a `MODELS` registry dict at the top of the file. Shared integration and sequential windowing logic lives in one `solve_and_window()` function. Run with `python solve_ode.py --model lotka_volterra` or `--model fitzhughnagumo`.
- **File:** `train_vae.py` *(replaces `train_vae_lotkavolt.py` and `train_vae_fitzhughnagumo.py`)* — The two training scripts were identical apart from data file, weights filename, and plot title. These are now config entries in a `MODELS` registry. Accepts `--epochs`, `--beta`, `--latent`, and `--anneal` flags for ad-hoc overrides without editing the file.
- **File:** `export_for_d3.py` *(replaces `export_for_d3.py` and `export_for_d3_fitzhughnagumo.py`)* — Unified export with per-model config for data file, weights, JSON/ONNX output paths, display alphas, and channel key names. The `ch1_key`/`ch2_key` config entries let FHN continue reusing `prey`/`predator` as JSON keys, preserving the `index.html` schema contract.
- **File:** `run_pipeline.sh` — Updated to call the three unified scripts with `--model` flags instead of the six separate scripts.
- **Adding a new ODE model** now only requires adding one dict entry to each of the three `MODELS` registries — no new files needed.

## 13. Lotka-Volterra γ Variant
- Added a third Lotka-Volterra view that varies γ (predator death rate) while fixing α=0.4, β=δ=0.02.
- γ ranges over `[0.1, 0.8]` with 20 linearly spaced values, matching the α range for direct comparability.
- **Note:** γ and α produce qualitatively similar latent spaces because both appear linearly in the oscillation frequency ω=√(αγ) and the fixed-point equations, so the VAE learns the same one-dimensional manifold traversed in the same direction. The view is retained for pedagogical comparison.
- **File:** `solve_odes.py` — Added `_lv_gamma_rhs` (fixes α, varies γ) and `lotka_volterra_gamma` registry entry outputting `lv_gamma_data.pt`.
- **File:** `train_vae.py` — Added `lotka_volterra_gamma` registry entry (`vae_lv_gamma.pt`).
- **File:** `export_for_d3.py` — Added `lotka_volterra_gamma` registry entry; display gammas `[0.1, 0.3, 0.5, 0.7]`; outputs `lv_gamma_data_for_d3.json` + `lv_gamma_decoder.onnx`.
- **File:** `index.html` — Added "Lotka-Volterra (γ)" switcher button with a purple colour ramp (hsl 260°) to distinguish it from the red α ramp.
- **File:** `run_pipeline.sh` — Added `lotka_volterra_gamma` to all three pipeline steps.

## 14. Lotka-Volterra β Variant
- Added a fourth view that varies β (predation rate) only, with δ fixed at 0.02. β ranges over `[0.01, 0.1]` with 20 linearly spaced values.
- At low β (≈0.01) prey populations swing to large spiky excursions; at high β (≈0.1) oscillations are tight and mild — a qualitatively different waveform shape compared to the α and γ variants, which only scale frequency.
- **File:** `solve_odes.py` — Added `_lv_beta_rhs` (fixes α=0.4, δ=0.02, γ=0.4, varies β only) and `lotka_volterra_beta` registry entry outputting `lv_beta_data.pt`.
- **File:** `train_vae.py` — Added `lotka_volterra_beta` registry entry (`vae_lv_beta.pt`).
- **File:** `export_for_d3.py` — Added `lotka_volterra_beta` registry entry; display betas `[0.01, 0.03, 0.06, 0.1]` spanning the full range; outputs `lv_beta_data_for_d3.json` + `lv_beta_decoder.onnx`.
- **File:** `index.html` — Added "Lotka-Volterra (β)" switcher button with a green colour ramp (hsl 130°); switcher now has four buttons with corrected border-radius CSS.
- **File:** `run_pipeline.sh` — Added `lotka_volterra_beta` to all three pipeline steps.

## 15. Equation Display with Highlighted Varied Parameter
- **File:** `index.html` — Added a MathJax-rendered equation panel between the switcher bar and the plots. Each model shows its governing equations with the varied parameter highlighted in gold (`{\color{#ffd166}{...}}`), scoped correctly using TeX grouping braces. Fixed parameter values are listed below the equations. MathJax is re-typeset on every model switch via `MathJax.typesetPromise`.
- FitzHugh-Nagumo parameter corrected from `α` to `a` throughout (title, legend, equation panel, JS config).
- FHN colour ramp changed from red to teal (hsl 185°) to visually distinguish it from Lotka-Volterra α.


## 16. FitzHugh-Nagumo I Variant
- Added a fifth model that varies I (external drive current) in the FitzHugh-Nagumo system, the canonical bifurcation parameter in neuroscience. I ranges over `[0.35, 1.5]` — starting just above the Hopf bifurcation so all 20 training trajectories are in the oscillatory regime. Fixed parameters: a=0.7, b=0.8, τ=0.4.
- Unlike varying `a` (which crosses a bifurcation), this view encodes only spiking frequency and amplitude changes within the oscillatory regime, producing a cleaner 1-D latent manifold.
- **File:** `solve_odes.py` — Added `_fhn_I_rhs` (fixes a=0.7, b=0.8, τ=0.4, varies I) and `fitzhughnagumo_I` registry entry outputting `fhn_I_data.pt`.
- **File:** `train_vae.py` — Added `fitzhughnagumo_I` registry entry (`vae_fhn_I.pt`).
- **File:** `export_for_d3.py` — Added `fitzhughnagumo_I` registry entry; display values `[0.35, 0.6, 0.9, 1.2, 1.5]`; outputs `fhn_I_data_for_d3.json` + `fhn_I_decoder.onnx`.
- **File:** `index.html` — Added "FitzHugh-Nagumo (I)" switcher button with an orange colour ramp (hsl 30°); equation panel shows I highlighted in the `dv/dt` term.
- **File:** `run_pipeline.sh` — Added `fitzhughnagumo_I` to all three pipeline steps.

## 18. Fixed Output Paths in Unified Pipeline Scripts
- **File:** `src/solve_odes.py` — Added `ROOT = Path(__file__).resolve().parent.parent`, `DATA_DIR = ROOT / 'data'`, and `DATA_DIR.mkdir(parents=True, exist_ok=True)`. All five `out_file` registry values updated from bare filenames (e.g. `'predator_prey_data.pt'`) to `DATA_DIR / '<filename>'`. Datasets now always save to `data/` regardless of working directory.
- **File:** `src/train_vae.py` — Added `ROOT`, `DATA_DIR`, `MODELS_DIR`, and `FIGURES_DIR` path constants (mirroring the pattern in the old per-model scripts). All five registry entries updated: `data_file` now reads from `DATA_DIR/`, `weights` saves to `MODELS_DIR/`, and `loss_plot` saves to `FIGURES_DIR/`. `mkdir(parents=True, exist_ok=True)` called for both output directories.

## 19. Quantitative Evaluation
- New top-level `evaluation/` folder containing a single entry-point script and the artifacts it produces.
- Three metric families per model: (a) reconstruction MSE on a deterministic 20% held-out split *and* on freshly-integrated trajectories at parameter values placed halfway between every pair of training values (a true held-out test, since `train_vae.py` uses an unseeded random split); (b) latent→parameter regression R² via both linear regression (full data) and 5-NN regression (held-out split) — the gap between the two is itself diagnostic, since most models organize the parameter along a curved 1-D manifold inside the 2-D code; (c) behavioral labeling (fixed_point vs. limit_cycle) by measuring channel-1 amplitude in the last 80% of each window, plus silhouette score on 2-D latents when both classes are present.
- Reuses the model registries from `src/solve_odes.py` and `src/train_vae.py` so adding a new model requires no changes here.
- **Notable finding:** `fitzhughnagumo` (a-sweep) yields kNN R² ≈ 0.02, meaning the trained VAE's latent space does not organize by `a` — every other sweep gets R² ≥ 0.78. Worth investigating before final write-up.
- **File:** `evaluation/evaluate.py` — Loops over all models (or `--model <key>` for one), writes `evaluation/results.json`, `evaluation/results.md` (markdown table with definitions), and `evaluation/metrics_summary.png` (paired bar charts of reconstruction MSE and R²).
- **File:** `README.md` — Added `evaluation/` to the repo layout, an entry to the Files table, and a Quantitative evaluation usage section.
- **File:** `run_pipeline.sh` — Fixed broken `solve_odes.py` / `train_vae.py` invocations to use the `src/` prefix.
