import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from vae import VAE

# Load dataset
data = torch.load('predator_prey_data.pt', weights_only=True)
x, y = data['x'], data['y']
window_size = data['window_size']
input_width = x.shape[1]

print(f"Loaded {x.shape[0]} samples, input dim = {input_width}")

# Train/val split (80/20)
n = x.shape[0]
n_train = int(0.8 * n)
perm = torch.randperm(n)
x_train, y_train = x[perm[:n_train]], y[perm[:n_train]]
x_val, y_val = x[perm[n_train:]], y[perm[n_train:]]

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64, shuffle=False)

# Train
model = VAE(input_width=input_width, latent_dim=3, learning_rate=1e-3)
train_losses, val_losses = model.train_model(train_loader, val_loader, epochs=100, beta=1.0)

# Plot losses
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('VAE Training Loss')
plt.legend()
plt.tight_layout()
plt.savefig('vae_loss_curve.png', dpi=150)
plt.show()

# Save weights
torch.save(model.state_dict(), 'vae_lotkavolt.pt')
print("Saved model weights to vae_lotkavolt.pt")
