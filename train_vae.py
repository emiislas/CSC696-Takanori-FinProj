"""
train_vae.py  —  train and save a VAE for any registered ODE dataset.

Usage:
    python train_vae.py --model lotka_volterra
    python train_vae.py --model fitzhughnagumo
    python train_vae.py --model lotka_volterra --epochs 300 --beta 0.5
"""

import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from vae import VAE

# ── Model registry ────────────────────────────────────────────────────────────

MODELS = {
    'lotka_volterra': dict(
        data_file  = 'predator_prey_data.pt',
        weights    = 'vae_lotkavolt.pt',
        loss_plot  = 'vae_loss_curve.png',
        plot_title = 'VAE Training Loss — Lotka-Volterra',
    ),
    'fitzhughnagumo': dict(
        data_file  = 'fitzhughnagumo_data.pt',
        weights    = 'vae_fitzhughnagumo.pt',
        loss_plot  = 'vae_loss_curve_fitzhughnagumo.png',
        plot_title = 'VAE Training Loss — FitzHugh-Nagumo',
    ),
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
