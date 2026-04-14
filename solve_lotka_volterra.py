import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def ode_func(t, X, a):
    x1, x2 = X
    alpha = a
    beta = 0.02
    gamma = 0.4
    delta = 0.02
    dx1dt = alpha * x1 - beta * x1 * x2
    dx2dt = delta * x1 * x2 - gamma * x2
    return [dx1dt, dx2dt]

dt = 0.1
t_vals = np.arange(0, 100, dt)
y0 = [10, 10]

alphas = [0.1, 0.4, 0.6, 0.8]

X_all = []
for a in alphas:
    sol = solve_ivp(lambda t, y: ode_func(t, y, a),
                    (t_vals[0], t_vals[-1]), y0,
                    t_eval=t_vals, rtol=1e-10, atol=1e-10).y.T
    X_all.append(sol)

X_all = torch.tensor(np.array(X_all), dtype=torch.float32)  # (4, 1000, 2)

window_size = 350           # 100 time steps 
num_samples_per_traj = 50 # number of random windows to draw per trajectory

windows = []
labels = []
rng = np.random.default_rng()

for i, a in enumerate(alphas):
    traj = X_all[i]                          # (1000, 2)
    max_start = traj.shape[0] - window_size  

    # draw random start indices (time-aligned for x1 and x2)
    starts = rng.integers(0, max_start + 1, size=num_samples_per_traj)

    for s in starts:
        w = traj[s : s + window_size, :]     # (window_size, 2) — both species, same time window
        windows.append(w.reshape(-1))        # flatten to (window_size*2,)
        labels.append(a)

x = torch.stack(windows)          # (n_alphas*num_samp_per, window_size*d)
y = torch.tensor(labels)          # (n_alphas*num_samp_per,)

# Normalize
x_mean = x.mean(dim=0)
x_std = x.std(dim=0) + 1e-8
x = (x - x_mean) / x_std

print(f"Dataset: {x.shape[0]} windows, input dim = {x.shape[1]}")

# Plot a few raw trajectories
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, a in enumerate(alphas):
    axes[0].plot(t_vals, X_all[i, :, 0].numpy(), label=rf"$\alpha$={a}")
    axes[1].plot(t_vals, X_all[i, :, 1].numpy(), label=rf"$\alpha$={a}")
axes[0].set_title(r"Prey $x_1$(t)")
axes[1].set_title(r"Predator $x_2$(t)")
for ax in axes:
    ax.set_xlabel("Time")
    ax.legend()
plt.tight_layout()
# plt.show()

# Save
torch.save({'x': x, 'y': y, 'x_mean': x_mean, 'x_std': x_std,
            'window_size': window_size}, 'predator_prey_data.pt')

print("Saved to predator_prey_data.pt")
