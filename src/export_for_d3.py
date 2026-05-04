"""
export_for_d3.py  —  encode a trained VAE dataset and export JSON + ONNX for the browser.

Usage:
    python src/export_for_d3.py --model lotka_volterra
    python src/export_for_d3.py --model fitzhughnagumo
"""

from pathlib import Path
import argparse
import json
import torch
import numpy as np
from vae import VAE

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
WEB_DIR = ROOT / 'web'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
WEB_DIR.mkdir(parents=True, exist_ok=True)

# ── Model registry ────────────────────────────────────────────────────────────

MODELS = {
    'lotka_volterra': dict(
        data_file      = DATA_DIR / 'predator_prey_data.pt',
        weights        = MODELS_DIR / 'vae_lotkavolt.pt',
        json_out       = WEB_DIR / 'data_for_d3.json',
        onnx_out       = MODELS_DIR / 'decoder.onnx',
        display_alphas = [0.1, 0.4, 0.6, 0.8],
        alpha_precision = 1,
        ch1_key        = 'prey',
        ch2_key        = 'predator',
    ),
    'lotka_volterra_beta': dict(
        data_file       = DATA_DIR / 'lv_beta_data.pt',
        weights         = MODELS_DIR / 'vae_lv_beta.pt',
        json_out        = WEB_DIR / 'lv_beta_data_for_d3.json',
        onnx_out        = MODELS_DIR / 'lv_beta_decoder.onnx',
        display_alphas  = [0.01, 0.03, 0.06, 0.1],
        alpha_precision = 2,
        ch1_key         = 'prey',
        ch2_key         = 'predator',
    ),
    'lotka_volterra_gamma': dict(
        data_file       = DATA_DIR / 'lv_gamma_data.pt',
        weights         = MODELS_DIR / 'vae_lv_gamma.pt',
        json_out        = WEB_DIR / 'lv_gamma_data_for_d3.json',
        onnx_out        = MODELS_DIR / 'lv_gamma_decoder.onnx',
        display_alphas  = [0.1, 0.3, 0.5, 0.7],
        alpha_precision = 1,
        ch1_key         = 'prey',
        ch2_key         = 'predator',
    ),
    'fitzhughnagumo': dict(
        data_file      = DATA_DIR / 'fitzhughnagumo_data.pt',
        weights        = MODELS_DIR / 'vae_fitzhughnagumo.pt',
        json_out       = WEB_DIR / 'fhn_data_for_d3.json',
        onnx_out       = MODELS_DIR / 'fhn_decoder.onnx',
        display_alphas = [0.1, 0.3, 0.5, 0.7],
        alpha_precision = 2,
        ch1_key        = 'prey',
        ch2_key        = 'predator',
    ),
    'fitzhughnagumo_I': dict(
        data_file       = DATA_DIR / 'fhn_I_data.pt',
        weights         = MODELS_DIR / 'vae_fhn_I.pt',
        json_out        = WEB_DIR / 'fhn_I_data_for_d3.json',
        onnx_out        = MODELS_DIR / 'fhn_I_decoder.onnx',
        display_alphas  = [0.35, 0.6, 0.9, 1.2, 1.5],
        alpha_precision = 2,
        ch1_key         = 'prey',
        ch2_key         = 'predator',
    ),
}

# ── Export logic ──────────────────────────────────────────────────────────────

def export(cfg, dt=0.1, latent_dim=2, z_padding=0.3):
    # Load dataset
    data = torch.load(cfg['data_file'], weights_only=True)
    x, y = data['x'], data['y']
    x_mean, x_std = data['x_mean'], data['x_std']
    window_size = data['window_size']
    input_width = x.shape[1]

    # Load model
    model = VAE(input_width=input_width, latent_dim=latent_dim, learning_rate=1e-3)
    model.load_state_dict(torch.load(cfg['weights'], weights_only=True))
    model.eval()

    # Encode all samples
    with torch.no_grad():
        mu, _ = model.encode(x)
    z       = mu.numpy()
    alphas  = y.numpy()

    def denorm(flat):
        return (flat * x_std + x_mean).numpy()

    # Map each training alpha to its nearest display alpha
    display_alphas = cfg['display_alphas']
    prec           = cfg['alpha_precision']
    all_train_alphas = sorted(set(round(float(a), prec + 2) for a in alphas))
    train_to_display = {
        min(all_train_alphas, key=lambda a, da=da: abs(a - da)): da
        for da in display_alphas
    }

    # Build sample list
    samples = []
    for i in range(x.shape[0]):
        a = round(float(alphas[i]), prec + 2)
        if a not in train_to_display:
            continue
        raw = denorm(x[i])
        ch1 = raw[0::2].tolist()
        ch2 = raw[1::2].tolist()
        samples.append({
            'id':              len(samples),
            'alpha':           train_to_display[a],
            'z':               z[i].tolist(),
            cfg['ch1_key']:    ch1,
            cfg['ch2_key']:    ch2,
        })

    # Latent axis ranges
    z_min = z.min(axis=0)
    z_max = z.max(axis=0)

    # Export decoder to ONNX
    dummy_z = torch.zeros(1, latent_dim, dtype=torch.float32)
    torch.onnx.export(
        model.decoder, (dummy_z,), str(cfg['onnx_out']),
        input_names=['z'], output_names=['x_recon'],
        dynamic_axes={'z': {0: 'batch'}, 'x_recon': {0: 'batch'}},
        opset_version=14, dynamo=False,
    )
    print(f"Exported decoder to {cfg['onnx_out']}")

    output = {
        'samples': samples,
        'x_mean':  x_mean.tolist(),
        'x_std':   x_std.tolist(),
        'meta': {
            'window_size': int(window_size),
            'dt':          dt,
            'z1_range':    [float(z_min[0] - z_padding), float(z_max[0] + z_padding)],
            'z2_range':    [float(z_min[1] - z_padding), float(z_max[1] + z_padding)],
            'alphas':      display_alphas,
        },
    }

    with open(cfg['json_out'], 'w', encoding='utf-8') as f:
        json.dump(output, f)
    print(f"Exported {len(samples)} samples to {cfg['json_out']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODELS), required=True)
    args = parser.parse_args()
    export(MODELS[args.model])


if __name__ == '__main__':
    main()
