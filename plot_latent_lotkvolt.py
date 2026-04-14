import torch
import matplotlib.pyplot as plt
from vae import VAE

# Load dataset
data = torch.load('predator_prey_data.pt', weights_only=True)
x, y = data['x'], data['y']
input_width = x.shape[1]

# Load model
model = VAE(input_width=input_width, latent_dim=2, learning_rate=1e-3)
model.load_state_dict(torch.load('vae_lotkavolt.pt', weights_only=True))
model.eval()

# Encode all samples
with torch.no_grad():
    mu, _ = model.encode(x)

z = mu.numpy()
alphas = y.numpy()

# 2D plot
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(z[:, 0], z[:, 1], c=alphas, cmap='viridis', edgecolors='k', linewidths=0.5)
ax.set_xlabel('$z_1$')
ax.set_ylabel('$z_2$')
ax.set_title('2D Latent Space Colored by α')
fig.colorbar(sc, ax=ax, label=r'$\alpha$', shrink=0.6)
plt.tight_layout()
plt.show()