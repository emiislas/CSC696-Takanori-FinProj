# Possible Improvements for VAE Interpolation Quality

## Current Setup
- 20 alpha values evenly spaced in [0.1, 0.8], 50 random windows each = 1,000 total samples
- VAE: 700 -> 256 -> 128 -> 64 -> 2 latent -> 64 -> 128 -> 256 -> 700
- Beta annealing: linear ramp from 0 to target over first 100 of 500 epochs
- Decoder exported to ONNX and run client-side via onnxruntime-web for arbitrary latent points
- Decoder has no physics constraints (outputs raw 700-dim vector)

## Remaining Options

### 1. Physics-Informed Loss
Add a regularization term that numerically differentiates the decoded trajectory and penalizes deviation from the Lotka-Volterra equations. This pushes decoded outputs toward physically valid dynamics even at unseen latent points.

### 2. Neural ODE Decoder (previously attempted, reverted)
Replace the MLP decoder with one that outputs ODE parameters (alpha, beta, gamma, delta, initial conditions) and integrates the actual Lotka-Volterra equations via `solve_ivp`. Every decoded trajectory is a valid ODE solution by construction. Most principled but the biggest code change.
