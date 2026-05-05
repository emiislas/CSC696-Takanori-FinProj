# VAE Quantitative Evaluation

| Model | n | Recon MSE (norm) | Generalization MSE (norm) | R² linear | R² kNN | Silhouette | Behavioral labels |
|---|---|---|---|---|---|---|---|
| `lotka_volterra` | 800 | 0.0359 | 0.0298 | 0.019 | 0.989 | N/A | fp=0, lc=800 |
| `lotka_volterra_beta` | 800 | 0.0677 | 0.0537 | 0.015 | 0.783 | N/A | fp=0, lc=800 |
| `lotka_volterra_gamma` | 800 | 0.0392 | 0.0341 | 0.033 | 0.989 | N/A | fp=0, lc=800 |
| `fitzhughnagumo` | 800 | 0.1216 | 0.1080 | 0.001 | 0.017 | N/A | fp=0, lc=800 |
| `fitzhughnagumo_I` | 800 | 0.0779 | 0.0830 | 0.549 | 0.952 | 0.119 | fp=142, lc=658 |

**Recon MSE (norm):** mean squared error on a deterministic 20% held-out subset of the existing windows, in normalized units. Caveat: training uses an unseeded random split, so some held-out windows may have been seen during training — use the generalization column for a true held-out estimate.

**Generalization MSE (norm):** windows integrated from fresh ODE trajectories at parameter values placed halfway between every pair of training values. These trajectories are guaranteed unseen.

**R² linear / R² kNN:** how well the swept parameter can be recovered from the 2D latent code by linear regression (full-data fit) and by kNN regression (5-NN, evaluated on the deterministic held-out split). A value near 1 means the latent space organizes cleanly by the parameter.

**Silhouette:** silhouette score of 2D latents using behavioral labels (fixed_point vs limit_cycle). Labels are assigned by measuring the amplitude (½(max-min)) of channel 1 in the last 80% of each window; amplitude below 0.05 (raw units) → fixed_point, otherwise limit_cycle. Marked N/A when only one class is present.
