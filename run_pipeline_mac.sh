#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Step 1: Solving ODEs ==="
python3 src/solve_odes.py --model lotka_volterra
python3 src/solve_odes.py --model lotka_volterra_gamma
python3 src/solve_odes.py --model lotka_volterra_beta
python3 src/solve_odes.py --model fitzhughnagumo
python3 src/solve_odes.py --model fitzhughnagumo_I

echo "=== Step 2: Training VAEs ==="
python3 src/train_vae.py --model lotka_volterra --beta 0.01 --anneal 50
python3 src/train_vae.py --model lotka_volterra_gamma --beta 0.01 --anneal 50
python3 src/train_vae.py --model lotka_volterra_beta --beta 0.01 --anneal 50
python3 src/train_vae.py --model fitzhughnagumo --beta 0.01 --anneal 50
python3 src/train_vae.py --model fitzhughnagumo_I --beta 0.01 --anneal 50

echo "=== Step 3: Exporting data for D3 ==="
python3 src/export_for_d3.py --model lotka_volterra
python3 src/export_for_d3.py --model lotka_volterra_gamma
python3 src/export_for_d3.py --model lotka_volterra_beta
python3 src/export_for_d3.py --model fitzhughnagumo
python3 src/export_for_d3.py --model fitzhughnagumo_I

echo "=== Step 4: Launching visualization ==="
echo "Open http://localhost:8000/web/index.html in your browser"
python3 -m http.server 8000
