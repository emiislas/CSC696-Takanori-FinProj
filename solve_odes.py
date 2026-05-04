"""
solve_ode.py  —  solve and save windowed trajectory data for any registered ODE model.

Usage:
    python solve_ode.py --model lotka_volterra
    python solve_ode.py --model fitzhughnagumo
"""

import argparse
import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ── Model registry ────────────────────────────────────────────────────────────
# Each entry defines everything needed to integrate, window, and save one model.

def _lv_rhs(t, X, a):
    x1, x2 = X
    return [a * x1 - 0.02 * x1 * x2,
            0.02 * x1 * x2 - 0.4 * x2]

def _lv_gamma_rhs(t, X, gamma):
    x1, x2 = X
    return [0.4 * x1 - 0.02 * x1 * x2,
            0.02 * x1 * x2 - gamma * x2]

def _lv_beta_rhs(t, X, beta):
    # delta tied to beta so equilibria stay fixed at (gamma/beta, alpha/beta) = (1,1)
    # only the cycle amplitude/shape changes
    x1, x2 = X
    return [0.4 * x1 - beta * x1 * x2,
            beta * x1 * x2 - 0.4 * x2]

def _fhn_rhs(t, X, a):
    v, w = X
    b, tau, I = 0.8, 0.4, 0.5
    return [v - (v**3) / 3 - w + I,
            tau * (v + a - b * w)]

MODELS = {
    'lotka_volterra': dict(
        rhs         = _lv_rhs,
        y0          = [10.0, 10.0],
        alphas      = np.linspace(0.1, 0.8, 20).tolist(),
        ch_names    = ['Prey', 'Predator'],
        out_file    = 'predator_prey_data.pt',
        plot_title  = (r'Prey $x_1$(t)', r'Predator $x_2$(t)'),
        param_label = r'$\alpha$',
    ),
    'lotka_volterra_beta': dict(
        rhs         = _lv_beta_rhs,
        y0          = [10.0, 10.0],
        alphas      = np.linspace(0.01, 0.06, 20).tolist(),   # beta (=delta) values
        ch_names    = ['Prey', 'Predator'],
        out_file    = 'lv_beta_data.pt',
        plot_title  = (r'Prey $x_1$(t)', r'Predator $x_2$(t)'),
        param_label = r'$\beta$',
    ),
    'lotka_volterra_gamma': dict(
        rhs         = _lv_gamma_rhs,
        y0          = [10.0, 10.0],
        alphas      = np.linspace(0.1, 0.8, 20).tolist(),   # gamma values
        ch_names    = ['Prey', 'Predator'],
        out_file    = 'lv_gamma_data.pt',
        plot_title  = (r'Prey $x_1$(t)', r'Predator $x_2$(t)'),
        param_label = r'$\gamma$',
    ),
    'fitzhughnagumo': dict(
        rhs         = _fhn_rhs,
        y0          = [0.0, 0.0],
        alphas      = np.linspace(0.10, 0.70, 20).tolist(),
        ch_names    = ['Voltage', 'Recovery'],
        out_file    = 'fitzhughnagumo_data.pt',
        plot_title  = (r'Membrane voltage $v$(t)', r'Recovery variable $w$(t)'),
        param_label = 'a',
    ),
}

# ── Shared integration & windowing ───────────────────────────────────────────

def solve_and_window(cfg,
                     dt=0.1, t_end=100,
                     window_size=350, num_windows=40, shift_pct=0.04):
    t_vals = np.arange(0, t_end, dt)

    # Integrate all trajectories
    X_all = []
    for a in cfg['alphas']:
        sol = solve_ivp(
            lambda t, y, a=a: cfg['rhs'](t, y, a),
            (t_vals[0], t_vals[-1]), cfg['y0'],
            t_eval=t_vals, rtol=1e-10, atol=1e-10,
        ).y.T
        X_all.append(sol)
    X_all = torch.tensor(np.array(X_all), dtype=torch.float32)  # (n_alpha, T, 2)

    # Sequential windowing
    windows, labels = [], []
    for i, a in enumerate(cfg['alphas']):
        traj = X_all[i]
        for j in range(num_windows):
            start = int(j * shift_pct * window_size)
            w = traj[start: start + window_size, :]
            windows.append(w.reshape(-1))
            labels.append(a)

    x = torch.stack(windows)
    y = torch.tensor(labels)

    # Normalize
    x_mean = x.mean(dim=0)
    x_std  = x.std(dim=0) + 1e-8
    x_norm = (x - x_mean) / x_std

    print(f"[{cfg['out_file']}] {x.shape[0]} windows, input dim = {x.shape[1]}")

    # Plot raw trajectories
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, a in enumerate(cfg['alphas']):
        axes[0].plot(t_vals, X_all[i, :, 0].numpy(),
                     label=f"{cfg['param_label']}={a:.2f}")
        axes[1].plot(t_vals, X_all[i, :, 1].numpy(),
                     label=f"{cfg['param_label']}={a:.2f}")
    for ax, title in zip(axes, cfg['plot_title']):
        ax.set_title(title)
        ax.set_xlabel('Time')
    plt.tight_layout()

    # Save
    torch.save(dict(x=x_norm, y=y, x_mean=x_mean, x_std=x_std,
                    window_size=window_size),
               cfg['out_file'])
    print(f"Saved to {cfg['out_file']}")

    return X_all, t_vals


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODELS), required=True,
                        help='Which ODE model to solve')
    args = parser.parse_args()
    solve_and_window(MODELS[args.model])


if __name__ == '__main__':
    main()
