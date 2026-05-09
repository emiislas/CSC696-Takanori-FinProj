#!/bin/bash
set -e

cd "$(dirname "$0")"

MODELS=(
  lotka_volterra lotka_volterra_gamma lotka_volterra_beta
  fitzhughnagumo fitzhughnagumo_I
  duffing_delta duffing_alpha duffing_beta duffing_gamma duffing_omega
  vdp_mu vdp_omega0 vdp_gamma vdp_omegaf
  lorenz_sigma lorenz_rho lorenz_beta
  brusselator_A brusselator_B
  selkov_a selkov_b
)

echo "=== Step 1: Solving ODEs ==="
for m in "${MODELS[@]}"; do
  python3 src/solve_odes.py --model "$m"
done

echo "=== Step 2: Training VAEs ==="
for m in "${MODELS[@]}"; do
  python3 src/train_vae.py --model "$m" --beta 0.01 --anneal 50
done

echo "=== Step 3: Exporting data for D3 ==="
for m in "${MODELS[@]}"; do
  python3 src/export_for_d3.py --model "$m"
done

echo "=== Step 4: Launching visualization ==="
echo "Open http://localhost:8000/web/index.html in your browser"
python3 -m http.server 8000
