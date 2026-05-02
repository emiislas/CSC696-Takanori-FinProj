#!/bin/bash
set -e

echo "=== Step 1a: Solving Lotka-Volterra ODE ==="
python3 solve_lotka_volterra.py

echo "=== Step 1b: Solving FitzHugh-Nagumo ODE ==="
python3 solve_fitzhughnagumo.py

echo "=== Step 2a: Training VAE (Lotka-Volterra) ==="
python3 train_vae_lotkavolt.py

echo "=== Step 2b: Training VAE (FitzHugh-Nagumo) ==="
python3 train_vae_fitzhughnagumo.py

echo "=== Step 3a: Exporting data for D3 (Lotka-Volterra) ==="
python3 export_for_d3.py

echo "=== Step 3b: Exporting data for D3 (FitzHugh-Nagumo) ==="
python3 export_for_d3_fitzhughnagumo.py

echo "=== Done. Open http://localhost:8000 ==="
python3 -m http.server
