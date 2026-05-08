"""
train_vae.py  —  train and save a VAE for any registered ODE dataset.

Usage:
    python train_vae.py --model lotka_volterra
    python train_vae.py --model fitzhughnagumo
    python train_vae.py --model lotka_volterra --epochs 300 --beta 0.5
"""

import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from vae import VAE

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / 'data'
MODELS_DIR  = ROOT / 'models'
FIGURES_DIR = ROOT / 'figures'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Model registry ────────────────────────────────────────────────────────────

MODELS = {
    'lotka_volterra': dict(
        data_file  = DATA_DIR / 'predator_prey_data.pt',
        weights    = MODELS_DIR / 'vae_lotkavolt.pt',
        loss_plot  = FIGURES_DIR / 'vae_loss_curve.png',
        plot_title = 'VAE Training Loss — Lotka-Volterra',
    ),
    'lotka_volterra_beta': dict(
        data_file  = DATA_DIR / 'lv_beta_data.pt',
        weights    = MODELS_DIR / 'vae_lv_beta.pt',
        loss_plot  = FIGURES_DIR / 'vae_loss_curve_lv_beta.png',
        plot_title = 'VAE Training Loss — Lotka-Volterra (β varies)',
    ),
    'lotka_volterra_gamma': dict(
        data_file  = DATA_DIR / 'lv_gamma_data.pt',
        weights    = MODELS_DIR / 'vae_lv_gamma.pt',
        loss_plot  = FIGURES_DIR / 'vae_loss_curve_lv_gamma.png',
        plot_title = 'VAE Training Loss — Lotka-Volterra (γ varies)',
    ),
    'fitzhughnagumo': dict(
        data_file  = DATA_DIR / 'fitzhughnagumo_data.pt',
        weights    = MODELS_DIR / 'vae_fitzhughnagumo.pt',
        loss_plot  = FIGURES_DIR / 'vae_loss_curve_fitzhughnagumo.png',
        plot_title = 'VAE Training Loss — FitzHugh-Nagumo',
    ),
    'fitzhughnagumo_I': dict(
        data_file  = DATA_DIR / 'fhn_I_data.pt',
        weights    = MODELS_DIR / 'vae_fhn_I.pt',
        loss_plot  = FIGURES_DIR / 'vae_loss_curve_fhn_I.png',
        plot_title = 'VAE Training Loss — FitzHugh-Nagumo (I varies)',
    ),

    # ── Duffing oscillator ────────────────────────────────────────────────
    'duffing_delta': dict(data_file=DATA_DIR/'duffing_delta_data.pt', weights=MODELS_DIR/'vae_duffing_delta.pt',
                          loss_plot=FIGURES_DIR/'vae_loss_duffing_delta.png',
                          plot_title='VAE Training Loss — Duffing (δ)'),
    'duffing_alpha': dict(data_file=DATA_DIR/'duffing_alpha_data.pt', weights=MODELS_DIR/'vae_duffing_alpha.pt',
                          loss_plot=FIGURES_DIR/'vae_loss_duffing_alpha.png',
                          plot_title='VAE Training Loss — Duffing (α)'),
    'duffing_beta':  dict(data_file=DATA_DIR/'duffing_beta_data.pt',  weights=MODELS_DIR/'vae_duffing_beta.pt',
                          loss_plot=FIGURES_DIR/'vae_loss_duffing_beta.png',
                          plot_title='VAE Training Loss — Duffing (β)'),
    'duffing_gamma': dict(data_file=DATA_DIR/'duffing_gamma_data.pt', weights=MODELS_DIR/'vae_duffing_gamma.pt',
                          loss_plot=FIGURES_DIR/'vae_loss_duffing_gamma.png',
                          plot_title='VAE Training Loss — Duffing (γ)'),
    'duffing_omega': dict(data_file=DATA_DIR/'duffing_omega_data.pt', weights=MODELS_DIR/'vae_duffing_omega.pt',
                          loss_plot=FIGURES_DIR/'vae_loss_duffing_omega.png',
                          plot_title='VAE Training Loss — Duffing (ω)'),

    # ── Forced Van der Pol ────────────────────────────────────────────────
    'vdp_mu':     dict(data_file=DATA_DIR/'vdp_mu_data.pt',     weights=MODELS_DIR/'vae_vdp_mu.pt',
                       loss_plot=FIGURES_DIR/'vae_loss_vdp_mu.png',
                       plot_title='VAE Training Loss — Van der Pol (μ)'),
    'vdp_omega0': dict(data_file=DATA_DIR/'vdp_omega0_data.pt', weights=MODELS_DIR/'vae_vdp_omega0.pt',
                       loss_plot=FIGURES_DIR/'vae_loss_vdp_omega0.png',
                       plot_title='VAE Training Loss — Van der Pol (ω₀)'),
    'vdp_gamma':  dict(data_file=DATA_DIR/'vdp_gamma_data.pt',  weights=MODELS_DIR/'vae_vdp_gamma.pt',
                       loss_plot=FIGURES_DIR/'vae_loss_vdp_gamma.png',
                       plot_title='VAE Training Loss — Van der Pol (γ)'),
    'vdp_omegaf': dict(data_file=DATA_DIR/'vdp_omegaf_data.pt', weights=MODELS_DIR/'vae_vdp_omegaf.pt',
                       loss_plot=FIGURES_DIR/'vae_loss_vdp_omegaf.png',
                       plot_title='VAE Training Loss — Van der Pol (ω_f)'),

    # ── Lorenz attractor (chaotic) ────────────────────────────────────────
    'lorenz_sigma': dict(data_file=DATA_DIR/'lorenz_sigma_data.pt', weights=MODELS_DIR/'vae_lorenz_sigma.pt',
                         loss_plot=FIGURES_DIR/'vae_loss_lorenz_sigma.png',
                         plot_title='VAE Training Loss — Lorenz (σ)'),
    'lorenz_rho':   dict(data_file=DATA_DIR/'lorenz_rho_data.pt',   weights=MODELS_DIR/'vae_lorenz_rho.pt',
                         loss_plot=FIGURES_DIR/'vae_loss_lorenz_rho.png',
                         plot_title='VAE Training Loss — Lorenz (ρ)'),
    'lorenz_beta':  dict(data_file=DATA_DIR/'lorenz_beta_data.pt',  weights=MODELS_DIR/'vae_lorenz_beta.pt',
                         loss_plot=FIGURES_DIR/'vae_loss_lorenz_beta.png',
                         plot_title='VAE Training Loss — Lorenz (β)'),

    # ── Brusselator ───────────────────────────────────────────────────────
    'brusselator_A': dict(data_file=DATA_DIR/'brusselator_A_data.pt', weights=MODELS_DIR/'vae_brusselator_A.pt',
                          loss_plot=FIGURES_DIR/'vae_loss_brusselator_A.png',
                          plot_title='VAE Training Loss — Brusselator (A)'),
    'brusselator_B': dict(data_file=DATA_DIR/'brusselator_B_data.pt', weights=MODELS_DIR/'vae_brusselator_B.pt',
                          loss_plot=FIGURES_DIR/'vae_loss_brusselator_B.png',
                          plot_title='VAE Training Loss — Brusselator (B)'),

    # ── Sel'kov glycolysis ────────────────────────────────────────────────
    'selkov_a': dict(data_file=DATA_DIR/'selkov_a_data.pt', weights=MODELS_DIR/'vae_selkov_a.pt',
                     loss_plot=FIGURES_DIR/'vae_loss_selkov_a.png',
                     plot_title='VAE Training Loss — Selkov (a)'),
    'selkov_b': dict(data_file=DATA_DIR/'selkov_b_data.pt', weights=MODELS_DIR/'vae_selkov_b.pt',
                     loss_plot=FIGURES_DIR/'vae_loss_selkov_b.png',
                     plot_title='VAE Training Loss — Selkov (b)'),
}

# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg, epochs=500, latent_dim=2, lr=1e-3, beta=1, beta_anneal_epochs=100):
    data = torch.load(cfg['data_file'], weights_only=True)
    x, y = data['x'], data['y']
    input_width = x.shape[1]
    print(f"Loaded {x.shape[0]} samples, input dim = {input_width}")

    # Train / val split (80/20)
    n = x.shape[0]
    perm = torch.randperm(n)
    split = int(0.8 * n)
    x_train, y_train = x[perm[:split]], y[perm[:split]]
    x_val,   y_val   = x[perm[split:]], y[perm[split:]]

    train_loader = DataLoader(TensorDataset(x_train, y_train),
                              batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(x_val, y_val),
                              batch_size=64, shuffle=False)

    model = VAE(input_width=input_width, latent_dim=latent_dim, learning_rate=lr)
    train_losses, val_losses = model.train_model(
        train_loader, val_loader,
        epochs=epochs, beta=beta, beta_anneal_epochs=beta_anneal_epochs,
    )

    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(cfg['plot_title'])
    plt.legend(); plt.tight_layout()
    plt.savefig(cfg['loss_plot'], dpi=150)
    plt.close()

    torch.save(model.state_dict(), cfg['weights'])
    print(f"Saved weights to {cfg['weights']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  choices=list(MODELS), required=True)
    parser.add_argument('--epochs', type=int,   default=500)
    parser.add_argument('--beta',   type=float, default=1.0)
    parser.add_argument('--latent', type=int,   default=2,
                        help='Latent space dimensionality')
    parser.add_argument('--anneal', type=int,   default=100,
                        help='Number of epochs over which to anneal beta from 0')
    args = parser.parse_args()
    train(MODELS[args.model],
          epochs=args.epochs, beta=args.beta,
          latent_dim=args.latent, beta_anneal_epochs=args.anneal)


if __name__ == '__main__':
    main()
