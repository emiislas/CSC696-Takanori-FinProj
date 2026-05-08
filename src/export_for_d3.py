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
# Channel count is inferred from the .pt file (n_channels field), so the same
# config block works for 2-channel and 3-channel systems.

def _cfg(stem, weights_name, json_name, onnx_name, display, prec):
    return dict(
        data_file=DATA_DIR / f'{stem}.pt',
        weights=MODELS_DIR / weights_name,
        json_out=WEB_DIR / json_name,
        onnx_out=MODELS_DIR / onnx_name,
        display_alphas=display,
        alpha_precision=prec,
    )

MODELS = {
    # ── Lotka-Volterra ─────────────────────────────────────────────────────
    'lotka_volterra': _cfg(
        'predator_prey_data', 'vae_lotkavolt.pt',
        'data_for_d3.json', 'decoder.onnx',
        [0.1, 0.4, 0.6, 0.8], 1),
    'lotka_volterra_beta': _cfg(
        'lv_beta_data', 'vae_lv_beta.pt',
        'lv_beta_data_for_d3.json', 'lv_beta_decoder.onnx',
        [0.01, 0.03, 0.06, 0.1], 2),
    'lotka_volterra_gamma': _cfg(
        'lv_gamma_data', 'vae_lv_gamma.pt',
        'lv_gamma_data_for_d3.json', 'lv_gamma_decoder.onnx',
        [0.1, 0.3, 0.5, 0.7], 1),

    # ── FitzHugh-Nagumo ────────────────────────────────────────────────────
    'fitzhughnagumo': _cfg(
        'fitzhughnagumo_data', 'vae_fitzhughnagumo.pt',
        'fhn_data_for_d3.json', 'fhn_decoder.onnx',
        [0.1, 0.3, 0.5, 0.7], 2),
    'fitzhughnagumo_I': _cfg(
        'fhn_I_data', 'vae_fhn_I.pt',
        'fhn_I_data_for_d3.json', 'fhn_I_decoder.onnx',
        [0.35, 0.6, 0.9, 1.2, 1.5], 2),

    # ── Duffing oscillator ────────────────────────────────────────────────
    'duffing_delta': _cfg('duffing_delta_data', 'vae_duffing_delta.pt',
                          'duffing_delta_data_for_d3.json', 'duffing_delta_decoder.onnx',
                          [0.05, 0.18, 0.32, 0.5], 2),
    'duffing_alpha': _cfg('duffing_alpha_data', 'vae_duffing_alpha.pt',
                          'duffing_alpha_data_for_d3.json', 'duffing_alpha_decoder.onnx',
                          [-1.5, -1.0, -0.5, 0.0, 0.5], 2),
    'duffing_beta':  _cfg('duffing_beta_data',  'vae_duffing_beta.pt',
                          'duffing_beta_data_for_d3.json', 'duffing_beta_decoder.onnx',
                          [0.5, 1.0, 1.5, 2.0], 2),
    'duffing_gamma': _cfg('duffing_gamma_data', 'vae_duffing_gamma.pt',
                          'duffing_gamma_data_for_d3.json', 'duffing_gamma_decoder.onnx',
                          [0.2, 0.35, 0.5, 0.7], 2),
    'duffing_omega': _cfg('duffing_omega_data', 'vae_duffing_omega.pt',
                          'duffing_omega_data_for_d3.json', 'duffing_omega_decoder.onnx',
                          [0.8, 1.0, 1.2, 1.4, 1.6], 2),

    # ── Forced Van der Pol ────────────────────────────────────────────────
    'vdp_mu':     _cfg('vdp_mu_data',     'vae_vdp_mu.pt',
                       'vdp_mu_data_for_d3.json', 'vdp_mu_decoder.onnx',
                       [0.2, 1.0, 2.0, 4.0], 2),
    'vdp_omega0': _cfg('vdp_omega0_data', 'vae_vdp_omega0.pt',
                       'vdp_omega0_data_for_d3.json', 'vdp_omega0_decoder.onnx',
                       [0.6, 0.9, 1.1, 1.4], 2),
    'vdp_gamma':  _cfg('vdp_gamma_data',  'vae_vdp_gamma.pt',
                       'vdp_gamma_data_for_d3.json', 'vdp_gamma_decoder.onnx',
                       [0.0, 0.8, 1.6, 2.5], 2),
    'vdp_omegaf': _cfg('vdp_omegaf_data', 'vae_vdp_omegaf.pt',
                       'vdp_omegaf_data_for_d3.json', 'vdp_omegaf_decoder.onnx',
                       [0.6, 1.0, 1.4, 2.0], 2),

    # ── Lorenz attractor (chaotic, 3-channel) ─────────────────────────────
    'lorenz_sigma': _cfg('lorenz_sigma_data', 'vae_lorenz_sigma.pt',
                         'lorenz_sigma_data_for_d3.json', 'lorenz_sigma_decoder.onnx',
                         [5.0, 10.0, 15.0, 20.0], 1),
    'lorenz_rho':   _cfg('lorenz_rho_data',   'vae_lorenz_rho.pt',
                         'lorenz_rho_data_for_d3.json', 'lorenz_rho_decoder.onnx',
                         [10.0, 20.0, 28.0, 35.0, 40.0], 1),
    'lorenz_beta':  _cfg('lorenz_beta_data',  'vae_lorenz_beta.pt',
                         'lorenz_beta_data_for_d3.json', 'lorenz_beta_decoder.onnx',
                         [1.5, 2.3, 3.0, 4.0], 2),

    # ── Brusselator ───────────────────────────────────────────────────────
    'brusselator_A': _cfg('brusselator_A_data', 'vae_brusselator_A.pt',
                          'brusselator_A_data_for_d3.json', 'brusselator_A_decoder.onnx',
                          [0.5, 1.0, 1.7, 2.5], 2),
    'brusselator_B': _cfg('brusselator_B_data', 'vae_brusselator_B.pt',
                          'brusselator_B_data_for_d3.json', 'brusselator_B_decoder.onnx',
                          [1.5, 2.5, 3.5, 4.5], 2),

    # ── Sel'kov glycolysis ────────────────────────────────────────────────
    'selkov_a': _cfg('selkov_a_data', 'vae_selkov_a.pt',
                     'selkov_a_data_for_d3.json', 'selkov_a_decoder.onnx',
                     [0.05, 0.1, 0.15, 0.2], 2),
    'selkov_b': _cfg('selkov_b_data', 'vae_selkov_b.pt',
                     'selkov_b_data_for_d3.json', 'selkov_b_decoder.onnx',
                     [0.4, 0.7, 1.0, 1.2], 2),
}

# ── Export logic ──────────────────────────────────────────────────────────────

def export(cfg, dt=0.1, latent_dim=2, z_padding=0.3):
    # Load dataset
    data = torch.load(cfg['data_file'], weights_only=True)
    x, y = data['x'], data['y']
    x_mean, x_std = data['x_mean'], data['x_std']
    window_size = int(data['window_size'])
    n_channels  = int(data.get('n_channels', 2))
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

    # Build sample list — channels stored as a list of arrays (channels[c] is
    # the c-th time series). De-interleaves the flat (t0_c0, t0_c1, ..., t0_cN,
    # t1_c0, ...) layout written by solve_odes.py.
    samples = []
    for i in range(x.shape[0]):
        a = round(float(alphas[i]), prec + 2)
        if a not in train_to_display:
            continue
        raw = denorm(x[i])
        channels = [raw[c::n_channels].tolist() for c in range(n_channels)]
        sample = {
            'id':       len(samples),
            'alpha':    train_to_display[a],
            'z':        z[i].tolist(),
            'channels': channels,
        }
        # Backward-compat aliases for 2-channel models so older browser code
        # that reads `prey` / `predator` keeps working.
        if n_channels == 2:
            sample['prey']     = channels[0]
            sample['predator'] = channels[1]
        samples.append(sample)

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
            'window_size': window_size,
            'n_channels':  n_channels,
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
