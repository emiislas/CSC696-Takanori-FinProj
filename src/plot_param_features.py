from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import correlate
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def lv(t, X, alpha, beta=0.02, gamma=0.4, delta=0.02):
    x1, x2 = X
    return [alpha * x1 - beta * x1 * x2,
            delta * x1 * x2 - gamma * x2]


def amplitude_and_phase(prey, predator, dt):
    # Drop transient — keep only the steady-state portion
    n = len(prey)
    skip = n // 5
    p = np.asarray(prey[skip:])
    q = np.asarray(predator[skip:])

    amp = 0.5 * (p.max() - p.min())

    # Period estimate from prey FFT (excluding DC component)
    spec = np.abs(np.fft.rfft(p - p.mean()))
    k_peak = 1 + int(np.argmax(spec[1:]))
    period_steps = len(p) / k_peak

    # Lag from cross-correlation, restricted to within ±half a period
    pc = p - p.mean()
    qc = q - q.mean()
    xcorr = correlate(qc, pc, mode='full')
    lags = np.arange(-len(p) + 1, len(p))
    half = period_steps / 2
    mask = (lags >= -half) & (lags <= half)
    best_lag = lags[mask][np.argmax(xcorr[mask])]

    phase = 2 * np.pi * best_lag / period_steps
    # Wrap to (-pi, pi]
    phase = (phase + np.pi) % (2 * np.pi) - np.pi
    return amp, phase


def main():
    dt = 0.1
    t_vals = np.arange(0, 200, dt)  # longer window for better period estimation
    y0 = [10.0, 10.0]
    alphas = np.linspace(0.1, 0.8, 50)

    amps, phases = [], []
    for a in alphas:
        sol = solve_ivp(lambda t, y: lv(t, y, a),
                        (t_vals[0], t_vals[-1]), y0,
                        t_eval=t_vals, rtol=1e-10, atol=1e-10).y
        prey, predator = sol[0], sol[1]
        amp, phase = amplitude_and_phase(prey, predator, dt)
        amps.append(amp)
        phases.append(phase)

    amps = np.array(amps)
    phases = np.array(phases)

    fig, (ax_amp, ax_phase) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_amp.plot(alphas, amps, 'o-', color='#06d6a0')
    ax_amp.set_xlabel(r'$\alpha$ (prey growth rate)')
    ax_amp.set_ylabel('Prey amplitude  (max − min) / 2')
    ax_amp.set_title('Parameter vs Amplitude')
    ax_amp.grid(alpha=0.3)

    ax_phase.plot(alphas, phases, 'o-', color='#ef476f')
    ax_phase.set_xlabel(r'$\alpha$ (prey growth rate)')
    ax_phase.set_ylabel('Phase lag predator − prey  (rad)')
    ax_phase.set_title('Parameter vs Phase Angle')
    ax_phase.axhline(np.pi / 2, ls='--', color='#888', lw=0.8, label=r'$\pi/2$')
    ax_phase.legend()
    ax_phase.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
