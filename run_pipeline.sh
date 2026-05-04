#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Step 1: Solving ODEs ==="
python src/solve_odes.py --model lotka_volterra
python src/solve_odes.py --model lotka_volterra_gamma
python src/solve_odes.py --model lotka_volterra_beta
python src/solve_odes.py --model fitzhughnagumo
python src/solve_odes.py --model fitzhughnagumo_I

echo "=== Step 2: Training VAEs ==="
python src/train_vae.py --model lotka_volterra --beta 0.01 --anneal 50
python src/train_vae.py --model lotka_volterra_gamma --beta 0.01 --anneal 50
python src/train_vae.py --model lotka_volterra_beta --beta 0.01 --anneal 50
python src/train_vae.py --model fitzhughnagumo --beta 0.01 --anneal 50
python src/train_vae.py --model fitzhughnagumo_I --beta 0.01 --anneal 50

echo "=== Step 3: Exporting data for D3 ==="
python src/export_for_d3.py --model lotka_volterra
python src/export_for_d3.py --model lotka_volterra_gamma
python src/export_for_d3.py --model lotka_volterra_beta
python src/export_for_d3.py --model fitzhughnagumo
python src/export_for_d3.py --model fitzhughnagumo_I

echo "=== Step 4: Launching visualization ==="
echo "Open http://localhost:8000/web/index.html in your browser"
python -m http.server 8000
