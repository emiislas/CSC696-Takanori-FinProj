from pathlib import Path
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

# Load dataset
data = torch.load(DATA_DIR / 'predator_prey_data.pt', weights_only=True)
x, y = data['x'], data['y']
x_mean, x_std = data['x_mean'], data['x_std']
window_size = data['window_size']
input_width = x.shape[1]

# Load model
model = VAE(input_width=input_width, latent_dim=2, learning_rate=1e-3)
model.load_state_dict(torch.load(MODELS_DIR / 'vae_lotkavolt.pt', weights_only=True))
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

# Only display samples whose training alpha is closest to one of these 4 values
display_alphas = [0.1, 0.4, 0.6, 0.8]
all_train_alphas = sorted(set(float(a) for a in alphas))
train_to_display = {}
for da in display_alphas:
    closest = min(all_train_alphas, key=lambda a, da=da: abs(a - da))
    train_to_display[closest] = da

# Build sample list (filtered to display alphas)
samples = []
for i in range(x.shape[0]):
    a = float(alphas[i])
    if a not in train_to_display:
        continue
    raw = denorm(x[i])  # (700,) interleaved: [prey0, pred0, prey1, pred1, ...]
    prey = raw[0::2].tolist()
    predator = raw[1::2].tolist()
    samples.append({
        'id': len(samples),
        'alpha': train_to_display[a],
        'z': z[i].tolist(),
        'prey': prey,
        'predator': predator,
    })

# Determine latent axis ranges from encoded data
z_min = z.min(axis=0)
z_max = z.max(axis=0)
padding = 0.3

# Export decoder to ONNX for client-side inference (Improvement 4a)
onnx_path = MODELS_DIR / 'decoder.onnx'
dummy_z = torch.zeros(1, 2, dtype=torch.float32)
torch.onnx.export(
    model.decoder,
    (dummy_z,),
    str(onnx_path),
    input_names=['z'],
    output_names=['x_recon'],
    dynamic_axes={'z': {0: 'batch'}, 'x_recon': {0: 'batch'}},
    opset_version=14,
    dynamo=False,
)
print(f"Exported decoder to {onnx_path}")

output = {
    'samples': samples,
    'x_mean': x_mean.tolist(),
    'x_std': x_std.tolist(),
    'meta': {
        'window_size': int(window_size),
        'dt': dt,
        'z1_range': [float(z_min[0] - padding), float(z_max[0] + padding)],
        'z2_range': [float(z_min[1] - padding), float(z_max[1] + padding)],
        'alphas': display_alphas,
    }
}

json_path = WEB_DIR / 'data_for_d3.json'
with open(json_path, 'w') as f:
    json.dump(output, f)

print(f"Exported {len(samples)} samples to {json_path}")
