import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def ode_func(t, X, a):
    v, w = X
    b, tau, I = 0.8, 0.4, 0.5
    dv = v - (v**3)/3 - w + I
    dw = tau * (v + a - b * w)
    return dv, dw

dt = 0.1
t_vals = np.arange(0, 100, dt)
y0 = [0, 0]

alphas = np.linspace(0.10, 0.7, 20).tolist()

X_all = []
for a in alphas:
    sol = solve_ivp(lambda t, y: ode_func(t, y, a),
                    (t_vals[0], t_vals[-1]), y0,
                    t_eval=t_vals, rtol=1e-10, atol=1e-10).y.T
    X_all.append(sol)

X_all = torch.tensor(np.array(X_all), dtype=torch.float32)  # (20, 1000, 2)

window_size = 350
num_windows_per_traj = 40
shift_percent = 0.04

windows = []
labels = []

for i, a in enumerate(alphas):
    traj = X_all[i]

    for j in range(num_windows_per_traj):
        start = int(j * (shift_percent * window_size))
        w = traj[start: start + window_size, :]  # (window_size, 2)
        windows.append(w.reshape(-1))             # flatten to (window_size*2,)
        labels.append(a)

x = torch.stack(windows)       # (n_alphas*num_windows, window_size*2)
y = torch.tensor(labels)       # (n_alphas*num_windows,)

# Normalize
x_mean = x.mean(dim=0)
x_std  = x.std(dim=0) + 1e-8
x = (x - x_mean) / x_std

print(f"Dataset: {x.shape[0]} windows, input dim = {x.shape[1]}")

# Plot a few raw trajectories
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, a in enumerate(alphas):
    axes[0].plot(t_vals, X_all[i, :, 0].numpy(), label=rf"$\alpha$={a:.2f}")
    axes[1].plot(t_vals, X_all[i, :, 1].numpy(), label=rf"$\alpha$={a:.2f}")
axes[0].set_title(r"Membrane voltage $v$(t)")
axes[1].set_title(r"Recovery variable $w$(t)")
for ax in axes:
    ax.set_xlabel("Time")
plt.tight_layout()
# plt.show()

# Save
torch.save({'x': x, 'y': y, 'x_mean': x_mean, 'x_std': x_std,
            'window_size': window_size}, 'fitzhughnagumo_data.pt')

print("Saved to fitzhughnagumo_data.pt")
