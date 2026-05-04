from pathlib import Path
import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

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
#    plt.plot(sol[:,0])
#    plt.plot(sol[:,1])
#    plt.show()

    
    X_all.append(sol)

X_all = torch.tensor(np.array(X_all), dtype=torch.float32) # (4, 1000, 2)

window_size = 350 # 100 time steps 
num_windows_per_traj = 40 # number of random windows to draw per trajectory
shift_percent = 0.04

windows = []
labels = []
rng = np.random.default_rng()

for i,a in enumerate(alphas):
    traj = X_all[i]  

    for j in range(num_windows_per_traj):
        
        start = int(j*(shift_percent * window_size))

        w = traj[start : start + window_size, :] # (window_size, 2) — both species, same time window
        windows.append(w.reshape(-1)) # flatten to (window_size*2,)
        labels.append(a)


x = torch.stack(windows) # (n_alphas*num_samp_per, window_size*d)
y = torch.tensor(labels) # (n_alphas*num_samp_per,)

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
axes[0].set_title(r"$x_1$(t)")
axes[1].set_title(r"$x_2$(t)")
for ax in axes:
    ax.set_xlabel("Time")
    ax.legend()
plt.tight_layout()
plt.show()

# Save
out_path = DATA_DIR / 'fitzhughnagumo_data.pt'
torch.save({'x': x, 'y': y, 'x_mean': x_mean, 'x_std': x_std,
            'window_size': window_size}, out_path)

print(f"Saved to {out_path}")
