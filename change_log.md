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

## 10. Sequential window sampling
- Changed data generation s.t. instead of sampling random windows per trajectory we sample sequential windows which results on more uniform points in the latent space.