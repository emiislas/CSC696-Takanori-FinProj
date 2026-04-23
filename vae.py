import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr

class VAE(torch.nn.Module):
    def __init__(self, input_width, latent_dim=3, learning_rate=1e-3):
        super(VAE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_width, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Tanh(),
        )
        self.fc_mu      = torch.nn.Linear(64, latent_dim)
        self.fc_log_var = torch.nn.Linear(64, latent_dim)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, input_width),
        )
        self.latent_dim = latent_dim
        self.alpha = learning_rate

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def vae_loss(self, x_recon, x, mu, log_var, beta=1.0):
        recon_loss = torch.mean((x_recon - x) ** 2)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss

    def train_model(self, train_loader, val_loader, epochs, beta=1.0, beta_anneal_epochs=0):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.alpha, weight_decay=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            if beta_anneal_epochs > 0 and epoch < beta_anneal_epochs:
                current_beta = beta * (epoch / beta_anneal_epochs)
            else:
                current_beta = beta
            self.train()
            epoch_loss = 0
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                x_recon, mu, log_var = self.forward(batch_x)
                loss, _, _ = self.vae_loss(x_recon, batch_x, mu, log_var, current_beta)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_train = epoch_loss / len(train_loader)
            train_losses.append(avg_train)

            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    x_recon, mu, log_var = self.forward(batch_x)
                    loss, _, _ = self.vae_loss(x_recon, batch_x, mu, log_var, current_beta)
                    val_loss += loss.item()
            avg_val = val_loss / len(val_loader)
            val_losses.append(avg_val)

            scheduler.step(avg_val)
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Train: {avg_train:.4f}, Val: {avg_val:.4f}")
        return train_losses, val_losses
