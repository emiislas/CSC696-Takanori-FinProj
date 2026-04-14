import json
import torch
import numpy as np
from vae import VAE

# Load dataset
data = torch.load('predator_prey_data.pt', weights_only=True)
x, y = data['x'], data['y']
x_mean, x_std = data['x_mean'], data['x_std']
window_size = data['window_size']
input_width = x.shape[1]

# Load model
model = VAE(input_width=input_width, latent_dim=2, learning_rate=1e-3)
model.load_state_dict(torch.load('vae_lotkavolt.pt', weights_only=True))
model.eval()

dt = 0.1

# Encode all samples
with torch.no_grad():
    mu, _ = model.encode(x)

z = mu.numpy()
alphas = y.numpy()

# Denormalize a flattened window back to original scale
def denorm(flat):
    return (flat * x_std + x_mean).numpy()

# Build sample list
samples = []
for i in range(x.shape[0]):
    raw = denorm(x[i])  # (700,) interleaved: [prey0, pred0, prey1, pred1, ...]
    prey = raw[0::2].tolist()
    predator = raw[1::2].tolist()
    samples.append({
        'id': i,
        'alpha': float(alphas[i]),
        'z': z[i].tolist(),
        'prey': prey,
        'predator': predator,
    })

# Determine latent axis ranges from encoded data
z_min = z.min(axis=0)
z_max = z.max(axis=0)
z_mean = z.mean(axis=0)
padding = 0.3

# Generate decoded grid over z1 x z2
grid_n = 20
z1_range = np.linspace(z_min[0] - padding, z_max[0] + padding, grid_n)
z2_range = np.linspace(z_min[1] - padding, z_max[1] + padding, grid_n)

grid = []
for z1 in z1_range:
    for z2 in z2_range:
        zpt = torch.tensor([[z1, z2]], dtype=torch.float32)
        with torch.no_grad():
            decoded = model.decode(zpt)[0]
        raw = denorm(decoded)
        prey = raw[0::2].tolist()
        predator = raw[1::2].tolist()
        grid.append({
            'z': [float(z1), float(z2)],
            'prey': prey,
            'predator': predator,
        })

output = {
    'samples': samples,
    'grid': grid,
    'meta': {
        'window_size': int(window_size),
        'dt': dt,
        'grid_n': grid_n,
        'z1_range': [float(z_min[0] - padding), float(z_max[0] + padding)],
        'z2_range': [float(z_min[1] - padding), float(z_max[1] + padding)],
        'alphas': sorted(set(float(a) for a in alphas)),
    }
}

with open('data_for_d3.json', 'w') as f:
    json.dump(output, f)

print(f"Exported {len(samples)} samples and {len(grid)} grid points to data_for_d3.json")
