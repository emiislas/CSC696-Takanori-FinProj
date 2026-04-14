#!/bin/bash
set -e

echo "=== Step 1: Solving Lotka-Volterra ODE ==="
python solve_lotka_volterra.py

echo "=== Step 2: Training VAE ==="
python train_vae_lotkavolt.py

echo "=== Step 3: Exporting data for D3 ==="
python export_for_d3.py

echo "=== Done. Open http://localhost:8000 ==="
