#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Step 1: Solving Lotka-Volterra ODE ==="
python src/solve_lotka_volterra.py

echo "=== Step 2: Training VAE ==="
python src/train_vae_lotkavolt.py

echo "=== Step 3: Exporting data for D3 ==="
python src/export_for_d3.py

echo "=== Step 4: Launching visualization ==="
echo "Open http://localhost:8000/web/index.html in your browser"
python -m http.server 8000
