"""
evaluate.py  —  quantitative evaluation of trained VAEs.

For each registered model this script computes:

  1. Reconstruction MSE on a deterministic held-out split of the existing windows
     (caveat: train_vae.py uses an unseeded random split, so some held-out
     windows may have been seen during training; use (2) for a true held-out test).

  2. Generalization MSE: integrate fresh ODE trajectories at parameter values
     interleaved between the training values (never seen by the model), window
     them the same way, normalize with the training mean/std, and report MSE.

  3. Latent -> parameter regression quality: linear R^2 and kNN R^2 of the
     swept parameter as a function of the 2D latent code.

  4. Behavioral labels (fixed_point vs limit_cycle), assigned by measuring
     steady-state amplitude in the last 20% of each trajectory. If at least two
     classes are present and each has multiple samples, the silhouette score on
     2D latents is reported; otherwise it is marked N/A.

Outputs (written under evaluation/):
    results.json          full structured results
    results.md            human-readable summary table
    metrics_summary.png   bar chart of the key metrics across models

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --model lotka_volterra
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from vae import VAE  # noqa: E402
from solve_odes import MODELS as ODE_MODELS  # noqa: E402
from train_vae import MODELS as VAE_MODELS  # noqa: E402

OUT_DIR = ROOT / 'evaluation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
HELDOUT_FRAC = 0.2
AMPLITUDE_FIXED_POINT_THRESHOLD = 0.05  # in raw, denormalized channel-1 units


# ── Behavioral labeling ──────────────────────────────────────────────────────

def behavioral_label(window_raw: np.ndarray) -> str:
    """Classify a single windowed trajectory as fixed_point or limit_cycle.

    `window_raw` is a flat (window_size * 2,) array of denormalized values,
    interleaved (ch1, ch2, ch1, ch2, ...).
    """
    ch1 = window_raw[0::2]
    n = len(ch1)
    skip = n // 5  # drop first 20% as transient
    tail = ch1[skip:] if n - skip >= 4 else ch1
    amp = 0.5 * (tail.max() - tail.min())
    return 'fixed_point' if amp < AMPLITUDE_FIXED_POINT_THRESHOLD else 'limit_cycle'


# ── Held-out integration at unseen parameter values ──────────────────────────

def integrate_unseen(ode_cfg: dict, dt: float = 0.1, t_end: int = 100,
                     window_size: int = 350, num_windows: int = 40,
                     shift_pct: float = 0.04):
    """Integrate fresh trajectories at parameter values interleaved between
    those used for training, then window them identically to solve_odes.py.

    Returns a (num_unseen_windows, window_size*2) array of *raw* (un-normalized)
    windows, plus the matching parameter labels.
    """
    train_params = sorted(ode_cfg['alphas'])
    # Place a new test point halfway between every adjacent pair of training
    # parameters. These values are never seen by the model.
    test_params = [
        0.5 * (train_params[i] + train_params[i + 1])
        for i in range(len(train_params) - 1)
    ]

    t_vals = np.arange(0, t_end, dt)
    windows, labels = [], []
    for p in test_params:
        sol = solve_ivp(
            lambda t, y, p=p: ode_cfg['rhs'](t, y, p),
            (t_vals[0], t_vals[-1]), ode_cfg['y0'],
            t_eval=t_vals, rtol=1e-10, atol=1e-10,
        ).y.T  # shape (T, 2)
        for j in range(num_windows):
            start = int(j * shift_pct * window_size)
            w = sol[start: start + window_size, :]
            windows.append(w.reshape(-1))
            labels.append(p)

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.float32)


# ── Per-model evaluation ─────────────────────────────────────────────────────

def evaluate_model(name: str) -> dict:
    ode_cfg = ODE_MODELS[name]
    vae_cfg = VAE_MODELS[name]

    print(f"\n=== {name} ===")
    data = torch.load(vae_cfg['data_file'], weights_only=True)
    x, y = data['x'], data['y']
    x_mean, x_std = data['x_mean'], data['x_std']
    window_size = int(data['window_size'])
    input_width = x.shape[1]

    model = VAE(input_width=input_width, latent_dim=2, learning_rate=1e-3)
    model.load_state_dict(torch.load(vae_cfg['weights'], weights_only=True))
    model.eval()

    # ---- Reconstruction on held-out (deterministic) split ------------------
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(x.shape[0], generator=g)
    n_test = int(HELDOUT_FRAC * x.shape[0])
    test_idx = perm[:n_test]
    x_test = x[test_idx]

    with torch.no_grad():
        x_recon_test, _, _ = model(x_test)
        recon_mse_norm = torch.mean((x_recon_test - x_test) ** 2).item()
        # MSE in raw (denormalized) units: undo (x - mu) / sigma so the number
        # is interpretable in the original variable scale.
        x_recon_raw = x_recon_test * x_std + x_mean
        x_test_raw = x_test * x_std + x_mean
        recon_mse_raw = torch.mean((x_recon_raw - x_test_raw) ** 2).item()

    # ---- Generalization to unseen parameter values -------------------------
    unseen_raw, unseen_params = integrate_unseen(ode_cfg, window_size=window_size)
    unseen_norm = (unseen_raw - x_mean.numpy()) / x_std.numpy()
    unseen_norm_t = torch.from_numpy(unseen_norm.astype(np.float32))

    with torch.no_grad():
        x_recon_unseen, _, _ = model(unseen_norm_t)
        gen_mse_norm = torch.mean((x_recon_unseen - unseen_norm_t) ** 2).item()
        x_recon_unseen_raw = x_recon_unseen * x_std + x_mean
        unseen_raw_t = torch.from_numpy(unseen_raw)
        gen_mse_raw = torch.mean((x_recon_unseen_raw - unseen_raw_t) ** 2).item()

    # ---- Latent codes for the full training set ----------------------------
    with torch.no_grad():
        mu_all, _ = model.encode(x)
    z = mu_all.numpy()
    params = y.numpy()

    # ---- Latent -> parameter regression -----------------------------------
    lin = LinearRegression().fit(z, params)
    r2_linear = float(lin.score(z, params))

    # 5-fold-style: simple holdout of the same deterministic split for kNN
    z_train = z[perm[n_test:].numpy()]
    p_train = params[perm[n_test:].numpy()]
    z_held = z[perm[:n_test].numpy()]
    p_held = params[perm[:n_test].numpy()]
    knn = KNeighborsRegressor(n_neighbors=min(5, len(z_train))).fit(z_train, p_train)
    r2_knn_held = float(knn.score(z_held, p_held))

    # ---- Behavioral labels and silhouette ----------------------------------
    raw_windows = (x * x_std + x_mean).numpy()
    labels = np.array([behavioral_label(w) for w in raw_windows])
    n_fixed = int((labels == 'fixed_point').sum())
    n_cycle = int((labels == 'limit_cycle').sum())

    if n_fixed >= 2 and n_cycle >= 2:
        sil = float(silhouette_score(z, labels))
        sil_status = 'ok'
    else:
        sil = None
        sil_status = f'degenerate (fixed_point={n_fixed}, limit_cycle={n_cycle})'

    result = {
        'model': name,
        'n_samples': int(x.shape[0]),
        'n_heldout': int(n_test),
        'n_unseen_param_windows': int(unseen_norm_t.shape[0]),
        'reconstruction_mse_normalized': recon_mse_norm,
        'reconstruction_mse_raw_units': recon_mse_raw,
        'generalization_mse_normalized': gen_mse_norm,
        'generalization_mse_raw_units': gen_mse_raw,
        'latent_to_param_r2_linear': r2_linear,
        'latent_to_param_r2_knn_heldout': r2_knn_held,
        'silhouette_score': sil,
        'silhouette_status': sil_status,
        'n_fixed_point': n_fixed,
        'n_limit_cycle': n_cycle,
        'param_label': ode_cfg['param_label'].strip('$\\'),
    }
    print(f"  recon MSE (norm)           : {recon_mse_norm:.4f}")
    print(f"  recon MSE (raw units)      : {recon_mse_raw:.4f}")
    print(f"  generalization MSE (norm)  : {gen_mse_norm:.4f}")
    print(f"  generalization MSE (raw)   : {gen_mse_raw:.4f}")
    print(f"  latent->param R^2 (linear) : {r2_linear:.4f}")
    print(f"  latent->param R^2 (kNN)    : {r2_knn_held:.4f}")
    print(f"  silhouette                 : {sil if sil is not None else sil_status}")
    print(f"  fixed_point / limit_cycle  : {n_fixed} / {n_cycle}")
    return result


# ── Reporting ────────────────────────────────────────────────────────────────

def write_markdown(results: list[dict], path: Path):
    lines = [
        '# VAE Quantitative Evaluation',
        '',
        '| Model | n | Recon MSE (norm) | Generalization MSE (norm) | R² linear | R² kNN | Silhouette | Behavioral labels |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for r in results:
        sil = f"{r['silhouette_score']:.3f}" if r['silhouette_score'] is not None else 'N/A'
        lines.append(
            f"| `{r['model']}` "
            f"| {r['n_samples']} "
            f"| {r['reconstruction_mse_normalized']:.4f} "
            f"| {r['generalization_mse_normalized']:.4f} "
            f"| {r['latent_to_param_r2_linear']:.3f} "
            f"| {r['latent_to_param_r2_knn_heldout']:.3f} "
            f"| {sil} "
            f"| fp={r['n_fixed_point']}, lc={r['n_limit_cycle']} |"
        )
    lines += [
        '',
        '**Recon MSE (norm):** mean squared error on a deterministic 20% '
        'held-out subset of the existing windows, in normalized units. '
        'Caveat: training uses an unseeded random split, so some held-out '
        'windows may have been seen during training — use the generalization '
        'column for a true held-out estimate.',
        '',
        '**Generalization MSE (norm):** windows integrated from fresh ODE '
        'trajectories at parameter values placed halfway between every pair '
        'of training values. These trajectories are guaranteed unseen.',
        '',
        '**R² linear / R² kNN:** how well the swept parameter can be recovered '
        'from the 2D latent code by linear regression (full-data fit) and by '
        'kNN regression (5-NN, evaluated on the deterministic held-out split). '
        'A value near 1 means the latent space organizes cleanly by the '
        'parameter.',
        '',
        '**Silhouette:** silhouette score of 2D latents using behavioral labels '
        '(fixed_point vs limit_cycle). Labels are assigned by measuring the '
        'amplitude (½(max-min)) of channel 1 in the last 80% of each window; '
        f'amplitude below {AMPLITUDE_FIXED_POINT_THRESHOLD} (raw units) → '
        'fixed_point, otherwise limit_cycle. Marked N/A when only one class '
        'is present.',
        '',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Wrote {path}")


def write_summary_plot(results: list[dict], path: Path):
    names = [r['model'] for r in results]
    recon = [r['reconstruction_mse_normalized'] for r in results]
    gen = [r['generalization_mse_normalized'] for r in results]
    r2_lin = [r['latent_to_param_r2_linear'] for r in results]
    r2_knn = [r['latent_to_param_r2_knn_heldout'] for r in results]

    fig, (ax_mse, ax_r2) = plt.subplots(1, 2, figsize=(13, 4.5))
    idx = np.arange(len(names))
    width = 0.38

    ax_mse.bar(idx - width / 2, recon, width, label='held-out (norm)', color='#118ab2')
    ax_mse.bar(idx + width / 2, gen, width, label='unseen-param (norm)', color='#ef476f')
    ax_mse.set_xticks(idx)
    ax_mse.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax_mse.set_ylabel('Reconstruction MSE')
    ax_mse.set_title('Reconstruction error')
    ax_mse.legend()
    ax_mse.grid(alpha=0.3, axis='y')

    ax_r2.bar(idx - width / 2, r2_lin, width, label='linear', color='#06d6a0')
    ax_r2.bar(idx + width / 2, r2_knn, width, label='kNN (held-out)', color='#ffd166')
    ax_r2.axhline(1.0, color='#888', lw=0.7, ls='--')
    ax_r2.set_xticks(idx)
    ax_r2.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax_r2.set_ylabel('R² (latent → parameter)')
    ax_r2.set_title('Latent space parameter fidelity')
    ax_r2.set_ylim(min(0, min(r2_lin + r2_knn) - 0.05), 1.05)
    ax_r2.legend()
    ax_r2.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=list(VAE_MODELS),
                        help='Evaluate a single model (default: all)')
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    names = [args.model] if args.model else list(VAE_MODELS)
    results = [evaluate_model(n) for n in names]

    json_path = OUT_DIR / 'results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {json_path}")

    write_markdown(results, OUT_DIR / 'results.md')
    write_summary_plot(results, OUT_DIR / 'metrics_summary.png')


if __name__ == '__main__':
    main()
