"""
solve_ode.py  —  solve and save windowed trajectory data for any registered ODE model.

Usage:
    python solve_ode.py --model lotka_volterra
    python solve_ode.py --model fitzhughnagumo
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

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
    # only beta (predation rate) varies; delta is fixed at 0.02
    x1, x2 = X
    return [0.4 * x1 - beta * x1 * x2,
            0.02 * x1 * x2 - 0.4 * x2]

def _fhn_rhs(t, X, a):
    v, w = X
    b, tau, I = 0.8, 0.4, 0.5
    return [v - (v**3) / 3 - w + I,
            tau * (v + a - b * w)]

def _fhn_I_rhs(t, X, I):
    v, w = X
    a, b, tau = 0.7, 0.8, 0.4
    return [v - (v**3) / 3 - w + I,
            tau * (v + a - b * w)]

# ── Duffing oscillator (forced) ─────────────────────────────────────────────
# x'' + delta x' + alpha x + beta x^3 = gamma cos(omega t)
_DUFFING_DEFAULTS = dict(delta=0.3, alpha=-1.0, beta=1.0, gamma=0.5, omega=1.2)

def _duffing_rhs(t, X, p, sweep):
    x, v = X
    d = {**_DUFFING_DEFAULTS, sweep: p}
    return [v,
            d['gamma'] * np.cos(d['omega'] * t)
            - d['delta'] * v - d['alpha'] * x - d['beta'] * x**3]

def _duffing_delta_rhs(t, X, p): return _duffing_rhs(t, X, p, 'delta')
def _duffing_alpha_rhs(t, X, p): return _duffing_rhs(t, X, p, 'alpha')
def _duffing_beta_rhs (t, X, p): return _duffing_rhs(t, X, p, 'beta')
def _duffing_gamma_rhs(t, X, p): return _duffing_rhs(t, X, p, 'gamma')
def _duffing_omega_rhs(t, X, p): return _duffing_rhs(t, X, p, 'omega')

# ── Forced Van der Pol ──────────────────────────────────────────────────────
# x'' - mu (1 - x^2) x' + omega0^2 x = gamma cos(omega_f t)
_VDP_DEFAULTS = dict(mu=2.0, omega0=1.0, gamma=1.0, omega_f=1.4)

def _vdp_rhs(t, X, p, sweep):
    x, v = X
    d = {**_VDP_DEFAULTS, sweep: p}
    return [v,
            d['mu'] * (1 - x**2) * v - d['omega0']**2 * x
            + d['gamma'] * np.cos(d['omega_f'] * t)]

def _vdp_mu_rhs    (t, X, p): return _vdp_rhs(t, X, p, 'mu')
def _vdp_omega0_rhs(t, X, p): return _vdp_rhs(t, X, p, 'omega0')
def _vdp_gamma_rhs (t, X, p): return _vdp_rhs(t, X, p, 'gamma')
def _vdp_omegaf_rhs(t, X, p): return _vdp_rhs(t, X, p, 'omega_f')

# ── Lorenz attractor (3-channel, chaotic) ───────────────────────────────────
_LORENZ_DEFAULTS = dict(sigma=10.0, rho=28.0, beta=8.0/3.0)

def _lorenz_rhs(t, X, p, sweep):
    x, y, z = X
    d = {**_LORENZ_DEFAULTS, sweep: p}
    return [d['sigma'] * (y - x),
            x * (d['rho'] - z) - y,
            x * y - d['beta'] * z]

def _lorenz_sigma_rhs(t, X, p): return _lorenz_rhs(t, X, p, 'sigma')
def _lorenz_rho_rhs  (t, X, p): return _lorenz_rhs(t, X, p, 'rho')
def _lorenz_beta_rhs (t, X, p): return _lorenz_rhs(t, X, p, 'beta')

# ── Brusselator ─────────────────────────────────────────────────────────────
_BRUSS_DEFAULTS = dict(A=1.0, B=3.0)

def _bruss_rhs(t, X, p, sweep):
    x, y = X
    d = {**_BRUSS_DEFAULTS, sweep: p}
    return [d['A'] + x**2 * y - (d['B'] + 1) * x,
            d['B'] * x - x**2 * y]

def _bruss_A_rhs(t, X, p): return _bruss_rhs(t, X, p, 'A')
def _bruss_B_rhs(t, X, p): return _bruss_rhs(t, X, p, 'B')

# ── Sel'kov glycolysis model ────────────────────────────────────────────────
_SELKOV_DEFAULTS = dict(a=0.1, b=0.6)

def _selkov_rhs(t, X, p, sweep):
    x, y = X
    d = {**_SELKOV_DEFAULTS, sweep: p}
    return [-x + d['a'] * y + x**2 * y,
            d['b'] - d['a'] * y - x**2 * y]

def _selkov_a_rhs(t, X, p): return _selkov_rhs(t, X, p, 'a')
def _selkov_b_rhs(t, X, p): return _selkov_rhs(t, X, p, 'b')

MODELS = {
    'lotka_volterra': dict(
        rhs         = _lv_rhs,
        y0          = [10.0, 10.0],
        alphas      = np.linspace(0.1, 0.8, 20).tolist(),
        ch_names    = ['Prey', 'Predator'],
        out_file    = DATA_DIR / 'predator_prey_data.pt',
        plot_title  = (r'Prey $x_1$(t)', r'Predator $x_2$(t)'),
        param_label = r'$\alpha$',
    ),
    'lotka_volterra_beta': dict(
        rhs         = _lv_beta_rhs,
        y0          = [10.0, 10.0],
        alphas      = np.linspace(0.01, 0.1, 20).tolist(),   # beta values
        ch_names    = ['Prey', 'Predator'],
        out_file    = DATA_DIR / 'lv_beta_data.pt',
        plot_title  = (r'Prey $x_1$(t)', r'Predator $x_2$(t)'),
        param_label = r'$\beta$',
    ),
    'lotka_volterra_gamma': dict(
        rhs         = _lv_gamma_rhs,
        y0          = [10.0, 10.0],
        alphas      = np.linspace(0.1, 0.8, 20).tolist(),   # gamma values
        ch_names    = ['Prey', 'Predator'],
        out_file    = DATA_DIR / 'lv_gamma_data.pt',
        plot_title  = (r'Prey $x_1$(t)', r'Predator $x_2$(t)'),
        param_label = r'$\gamma$',
    ),
    'fitzhughnagumo': dict(
        rhs         = _fhn_rhs,
        y0          = [0.0, 0.0],
        alphas      = np.linspace(0.10, 0.70, 20).tolist(),
        ch_names    = ['Voltage', 'Recovery'],
        out_file    = DATA_DIR / 'fitzhughnagumo_data.pt',
        plot_title  = (r'Membrane voltage $v$(t)', r'Recovery variable $w$(t)'),
        param_label = 'a',
    ),
    'fitzhughnagumo_I': dict(
        rhs         = _fhn_I_rhs,
        y0          = [0.0, 0.0],
        alphas      = np.linspace(0.35, 1.5, 20).tolist(),   # I values; 0.35 is just above Hopf
        ch_names    = ['Voltage', 'Recovery'],
        out_file    = DATA_DIR / 'fhn_I_data.pt',
        plot_title  = (r'Membrane voltage $v$(t)', r'Recovery variable $w$(t)'),
        param_label = 'I',
    ),

    # ── Duffing oscillator (forced) ───────────────────────────────────────
    'duffing_delta': dict(
        rhs=_duffing_delta_rhs, y0=[0.0, 0.0],
        alphas=np.linspace(0.05, 0.5, 20).tolist(),
        ch_names=['x', 'v'],
        out_file=DATA_DIR / 'duffing_delta_data.pt',
        plot_title=(r'Position $x(t)$', r'Velocity $v(t)$'),
        param_label=r'$\delta$',
    ),
    'duffing_alpha': dict(
        rhs=_duffing_alpha_rhs, y0=[0.0, 0.0],
        alphas=np.linspace(-1.5, 0.5, 20).tolist(),
        ch_names=['x', 'v'],
        out_file=DATA_DIR / 'duffing_alpha_data.pt',
        plot_title=(r'Position $x(t)$', r'Velocity $v(t)$'),
        param_label=r'$\alpha$',
    ),
    'duffing_beta': dict(
        rhs=_duffing_beta_rhs, y0=[0.0, 0.0],
        alphas=np.linspace(0.5, 2.0, 20).tolist(),
        ch_names=['x', 'v'],
        out_file=DATA_DIR / 'duffing_beta_data.pt',
        plot_title=(r'Position $x(t)$', r'Velocity $v(t)$'),
        param_label=r'$\beta$',
    ),
    'duffing_gamma': dict(
        rhs=_duffing_gamma_rhs, y0=[0.0, 0.0],
        alphas=np.linspace(0.2, 0.7, 20).tolist(),
        ch_names=['x', 'v'],
        out_file=DATA_DIR / 'duffing_gamma_data.pt',
        plot_title=(r'Position $x(t)$', r'Velocity $v(t)$'),
        param_label=r'$\gamma$',
    ),
    'duffing_omega': dict(
        rhs=_duffing_omega_rhs, y0=[0.0, 0.0],
        alphas=np.linspace(0.8, 1.6, 20).tolist(),
        ch_names=['x', 'v'],
        out_file=DATA_DIR / 'duffing_omega_data.pt',
        plot_title=(r'Position $x(t)$', r'Velocity $v(t)$'),
        param_label=r'$\omega$',
    ),

    # ── Forced Van der Pol ────────────────────────────────────────────────
    'vdp_mu': dict(
        rhs=_vdp_mu_rhs, y0=[0.5, 0.0],
        alphas=np.linspace(0.2, 4.0, 20).tolist(),
        ch_names=['x', 'v'],
        out_file=DATA_DIR / 'vdp_mu_data.pt',
        plot_title=(r'Position $x(t)$', r'Velocity $v(t)$'),
        param_label=r'$\mu$',
    ),
    'vdp_omega0': dict(
        rhs=_vdp_omega0_rhs, y0=[0.5, 0.0],
        alphas=np.linspace(0.6, 1.4, 20).tolist(),
        ch_names=['x', 'v'],
        out_file=DATA_DIR / 'vdp_omega0_data.pt',
        plot_title=(r'Position $x(t)$', r'Velocity $v(t)$'),
        param_label=r'$\omega_0$',
    ),
    'vdp_gamma': dict(
        rhs=_vdp_gamma_rhs, y0=[0.5, 0.0],
        alphas=np.linspace(0.0, 2.5, 20).tolist(),
        ch_names=['x', 'v'],
        out_file=DATA_DIR / 'vdp_gamma_data.pt',
        plot_title=(r'Position $x(t)$', r'Velocity $v(t)$'),
        param_label=r'$\gamma$',
    ),
    'vdp_omegaf': dict(
        rhs=_vdp_omegaf_rhs, y0=[0.5, 0.0],
        alphas=np.linspace(0.6, 2.0, 20).tolist(),
        ch_names=['x', 'v'],
        out_file=DATA_DIR / 'vdp_omegaf_data.pt',
        plot_title=(r'Position $x(t)$', r'Velocity $v(t)$'),
        param_label=r'$\omega_f$',
    ),

    # ── Lorenz attractor (3-channel, chaotic) ─────────────────────────────
    'lorenz_sigma': dict(
        rhs=_lorenz_sigma_rhs, y0=[1.0, 1.0, 1.0],
        alphas=np.linspace(5.0, 20.0, 20).tolist(),
        ch_names=['x', 'y', 'z'],
        out_file=DATA_DIR / 'lorenz_sigma_data.pt',
        plot_title=(r'$x(t)$', r'$y(t)$', r'$z(t)$'),
        param_label=r'$\sigma$',
    ),
    'lorenz_rho': dict(
        rhs=_lorenz_rho_rhs, y0=[1.0, 1.0, 1.0],
        alphas=np.linspace(10.0, 40.0, 20).tolist(),
        ch_names=['x', 'y', 'z'],
        out_file=DATA_DIR / 'lorenz_rho_data.pt',
        plot_title=(r'$x(t)$', r'$y(t)$', r'$z(t)$'),
        param_label=r'$\rho$',
    ),
    'lorenz_beta': dict(
        rhs=_lorenz_beta_rhs, y0=[1.0, 1.0, 1.0],
        alphas=np.linspace(1.5, 4.0, 20).tolist(),
        ch_names=['x', 'y', 'z'],
        out_file=DATA_DIR / 'lorenz_beta_data.pt',
        plot_title=(r'$x(t)$', r'$y(t)$', r'$z(t)$'),
        param_label=r'$\beta$',
    ),

    # ── Brusselator ───────────────────────────────────────────────────────
    'brusselator_A': dict(
        rhs=_bruss_A_rhs, y0=[1.0, 1.0],
        alphas=np.linspace(0.5, 2.5, 20).tolist(),
        ch_names=['x', 'y'],
        out_file=DATA_DIR / 'brusselator_A_data.pt',
        plot_title=(r'$x(t)$', r'$y(t)$'),
        param_label='A',
    ),
    'brusselator_B': dict(
        rhs=_bruss_B_rhs, y0=[1.0, 1.0],
        alphas=np.linspace(1.5, 4.5, 20).tolist(),
        ch_names=['x', 'y'],
        out_file=DATA_DIR / 'brusselator_B_data.pt',
        plot_title=(r'$x(t)$', r'$y(t)$'),
        param_label='B',
    ),

    # ── Sel'kov glycolysis ────────────────────────────────────────────────
    'selkov_a': dict(
        rhs=_selkov_a_rhs, y0=[1.0, 1.0],
        alphas=np.linspace(0.05, 0.2, 20).tolist(),
        ch_names=['x', 'y'],
        out_file=DATA_DIR / 'selkov_a_data.pt',
        plot_title=(r'$x(t)$', r'$y(t)$'),
        param_label='a',
    ),
    'selkov_b': dict(
        rhs=_selkov_b_rhs, y0=[1.0, 1.0],
        alphas=np.linspace(0.4, 1.2, 20).tolist(),
        ch_names=['x', 'y'],
        out_file=DATA_DIR / 'selkov_b_data.pt',
        plot_title=(r'$x(t)$', r'$y(t)$'),
        param_label='b',
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
    X_all = torch.tensor(np.array(X_all), dtype=torch.float32)  # (n_alpha, T, n_channels)
    n_channels = X_all.shape[2]

    # Sequential windowing — flatten interleaves channels per timestep:
    # (t0_c0, t0_c1, ..., t0_cN, t1_c0, ...)
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

    print(f"[{cfg['out_file']}] {x.shape[0]} windows, "
          f"input dim = {x.shape[1]}, channels = {n_channels}")

    # Plot raw trajectories — one subplot per channel
    fig, axes = plt.subplots(1, n_channels, figsize=(6 * n_channels, 4))
    if n_channels == 1:
        axes = [axes]
    for i, a in enumerate(cfg['alphas']):
        for c in range(n_channels):
            axes[c].plot(t_vals, X_all[i, :, c].numpy(),
                         label=f"{cfg['param_label']}={a:.2f}")
    titles = cfg.get('plot_title', tuple(f'channel {c}' for c in range(n_channels)))
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel('Time')
    plt.tight_layout()

    # Save
    torch.save(dict(x=x_norm, y=y, x_mean=x_mean, x_std=x_std,
                    window_size=window_size, n_channels=n_channels),
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
