# Change Log

## 1. Local HTTP Server in Pipeline
- **File:** `run_pipeline.sh`
- Added Step 4 that starts `python -m http.server 8000` after data export, fixing CORS errors when opening `index.html` directly via `file://`.

## 2. Zoom on Latent Space Plot
- **File:** `index.html`
- Added D3 zoom behavior (scroll to zoom, drag to pan) on the latent space scatter plot.
- Added a clip path so dots stay within the plot area when zoomed.
- Saved axis group references (`xAxisG`, `yAxisG`) and wrapped dots in a clipped `<g>` for zoom redraws.

## 3. Created Improvements Roadmap
- **File:** `IMPROVEMENTS.md`
- Documented 6 possible improvements: more training data, physics-informed loss, Neural ODE decoder, beta annealing, deeper network, and continuous latent decoding.

## 4. More Training Data (Improvement 1)
- **File:** `solve_lotka_volterra.py`
- Changed from 4 hardcoded alpha values to 20 evenly spaced values via `np.linspace(0.1, 0.8, 20)`, giving 1,000 training samples instead of 200.

## 5. Beta Annealing (Improvement 4)
- **File:** `vae.py` — Added `beta_anneal_epochs` parameter to `VAE.train_model`. Beta linearly ramps from 0 to target over the specified number of epochs.
- **File:** `train_vae_lotkavolt.py` — Set `beta_anneal_epochs=100` (ramp over first 100 of 500 epochs).

## 6. Filtered Display Alphas
- **File:** `export_for_d3.py`
- Training uses 20 alphas, but visualization only shows samples from the 4 closest to the original values (0.1, 0.4, 0.6, 0.8). Keeps the scatter plot clean while benefiting from denser training coverage.

## 7. Dense Interpolation Grid (Improvement 6c)
- **File:** `export_for_d3.py`
- Increased grid from 20x20 to 100x100 (10,000 decode points) so clicking anywhere in the latent space snaps to a very nearby grid point.

## 8. Neural ODE Decoder (Improvement 3) — REVERTED
- Was implemented but reverted. The `NeuralODEVAE` class was removed from `vae.py` and the training/export scripts were restored to use the plain `VAE`. The approach remains documented in `IMPROVEMENTS.md` as a future option.
