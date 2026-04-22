# Possible Improvements for VAE Interpolation Quality

## Current Setup
- 4 alpha values (0.1, 0.4, 0.6, 0.8), 50 random windows each = 200 total samples
- Small VAE: 700 -> 128 -> 64 -> 2 latent -> 64 -> 128 -> 700
- 20x20 = 400 grid decode points
- Decoder has no physics constraints (outputs raw 700-dim vector)

## Improvement Options

### 1. More Training Data
Add more alpha values (e.g., 20 evenly spaced between 0.1-0.8 instead of just 4) and more windows per trajectory. The VAE currently learns from very sparse parameter coverage, so interpolation between the 4 clusters is mostly guesswork.

### 2. Physics-Informed Loss
Add a regularization term that numerically differentiates the decoded trajectory and penalizes deviation from the Lotka-Volterra equations. This pushes decoded outputs toward physically valid dynamics even at unseen latent points.

### 3. Neural ODE Decoder
Replace the MLP decoder with one that outputs ODE parameters (alpha, beta, gamma, delta, initial conditions) and integrates the actual Lotka-Volterra equations via `solve_ivp`. Every decoded trajectory is a valid ODE solution by construction. Most principled but the biggest code change.

### 4. Beta Annealing
Start training with beta=0 (pure autoencoder) and slowly ramp to beta=1 over epochs. This lets the network learn good reconstructions before the KL term compresses the latent space, often producing a smoother, more faithful latent space.

### 5. Deeper Network
The current 2-hidden-layer architecture is small for 700-dim input. Adding layers (e.g., 700 -> 256 -> 128 -> 64 -> 2) could improve reconstruction fidelity.

### 6. Continuous Latent Space Decoding (Click-Anywhere)
Currently the visualization snaps to the nearest pre-computed grid point when clicking empty space. To decode arbitrary latent points on the fly, three approaches are possible:

- **6a. ONNX Runtime Web** — Export the decoder to ONNX and run it client-side with onnxruntime-web. The decoder is small (2 -> 64 -> 128 -> 700) and would run instantly in the browser. No server needed, truly continuous.
- **6b. Local API server** — Run a small Flask/FastAPI server that takes a (z1, z2) request, runs the PyTorch decoder, and returns the trajectory via JSON. Simpler to implement but requires the Python server running.
- **6c. Dense grid** — Massively increase grid density (e.g., 100x100 = 10,000 points). Not truly continuous but snapping becomes imperceptible. Downside: large JSON file.
