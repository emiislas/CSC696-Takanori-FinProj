# Speaker Notes

Target total time: **5–7 minutes** across the four slides below.
Pacing target in parentheses next to each slide.

---

## Slide: Objective  *(~60 s)*

> "Most parameter studies of an ODE work the same way: pick a parameter,
> solve the system, look at the trajectory, change the parameter, solve
> again. That's fine for one or two parameters but it scales badly — and
> it never gives you a *picture* of how the family of solutions is
> organized."

Key points to say out loud:

- Our goal is a **visual analysis framework** that lets a user see how
  ODE solutions change *as a function of the parameter*, without solving
  the ODE again every time.
- We do this by training a Variational Autoencoder on windowed
  trajectories and using the **2-D latent space** as the parameter map.
- Why visual analytics: the latent space becomes an interactive canvas —
  hovering, clicking, and decoding turn a black-box model into something
  the user can directly explore. The "look up a trajectory by clicking
  in latent space" is itself a new analysis primitive.
- Frame it as a tool, not just a model: the deliverable is a browser
  app, not a notebook plot.

Transition: "To do that, we first need data."

---

## Slide: Data Generation  *(~75 s)*

> "We picked two ODE families that show up everywhere in dynamical
> systems courses, because the goal is to demonstrate the *method*, not
> to push a particular biological model."

Key points:

- **Lotka–Volterra** (predator–prey). We sweep three different
  parameters separately — α (prey growth), β (predation rate), and γ
  (predator death) — to see whether the latent space picks up the same
  structure for each. β is the most interesting because it changes the
  *shape* of the limit cycle, not just its frequency.
- **FitzHugh–Nagumo** (neuron model). We sweep two parameters — `a` (the
  recovery bias) and `I` (the external drive current). `I` is the
  textbook bifurcation parameter for this system, so we expect it to
  cross a Hopf bifurcation in our sweep range.
- Per parameter we integrate 20 trajectories with `scipy.solve_ivp`,
  then chop each into 40 short overlapping windows. That gives ~800
  training examples per model — enough to train a small VAE without
  needing GPUs.
- Each window is normalized per-feature; the normalization stats are
  saved alongside so we can de-normalize anything the decoder produces
  back to the original physical units.

Transition: "Once the VAE is trained, the visualization becomes the
interface."

---

## Slide: Visualization  *(~2 min)*

> "There are three coordinated views, and they're all alive at the same
> time."

Walk through them in order:

1. **Latent space view** (left).
   - 2-D scatter of every windowed sample. Colored by the swept
     parameter using a perceptually-ordered ramp.
   - This is the parameter map. The fact that there's *visible
     structure* — almost always a curve or arc — is the unsupervised
     discovery story: we never told the network what α was, but it
     organized the data along a one-dimensional manifold that tracks α.
   - You can click a dot or click anywhere in empty space.

2. **Trajectory view** (right, top).
   - Shows the prey and predator (or voltage and recovery) traces over
     time for whichever point you clicked. Two channels per system,
     stacked.
   - When you click a *training sample*, you see the original
     simulation. When you click *empty latent space*, the decoder
     synthesizes a trajectory for that latent code — a virtual
     simulation you've never run.

3. **Parameter feature view** (right, bottom — the new addition).
   - Two scatter plots: parameter vs. amplitude, and parameter vs.
     phase angle (predator–prey lag, computed from
     cross-correlation in the browser).
   - These are the dynamical features a domain user actually cares
     about. They give the latent space a *physical* axis to compare
     against.
   - All three views are linked: clicking anywhere highlights the
     matching point in all three simultaneously.

Other interactive hooks worth mentioning if time:
- Equation panel above the views, rendered with MathJax, with the swept
  parameter highlighted in gold. Updates instantly when you switch
  models.
- Five-button switcher swaps the dataset, the decoder ONNX, the color
  ramp, and the equation panel together.
- Decoding runs **client-side** via onnxruntime-web — no server
  round-trip, so dragging the crosshair feels real-time.

Transition: "So that's what the tool does. Did it work?"

---

## Slide: Evaluation  *(~2 min)*

> "We evaluated this two ways — qualitative case studies plus a
> quantitative evaluation pass."

**Qualitative / case studies** (what we found by *using* the tool):

- For Lotka–Volterra α and γ, the latent forms a clean 1-D arc — the
  parameter is recovered with kNN R² of 0.99. Expected, because both
  parameters only scale frequency.
- For Lotka–Volterra β, you can *see* the waveform shape change as you
  walk along the latent: low β gives spiky excursions, high β gives
  tight oscillations. The latent picks this up without supervision.
- For FitzHugh–Nagumo with current I, the latent splits into two visible
  blobs — fixed-point trajectories on one side, limit-cycle on the
  other. The Hopf bifurcation appears as a *spatial* feature in the
  latent.

**Qualitative feedback** from classmates:
- (Insert what your reviewers said. Common comments to expect: the
  switcher is intuitive, the equation panel helps grounding, the click-
  to-decode interaction is the most engaging part.)

**Quantitative evaluation** (the script in `evaluation/evaluate.py`):

- **Reconstruction MSE** measured two ways: on a held-out subset of
  existing windows (~0.04–0.12 in normalized units across the five
  models), and on freshly-integrated trajectories at parameter values
  *halfway between* every training value — a true held-out test. The
  generalization MSE is essentially the same as the held-out MSE for
  every model, which means the VAE interpolates cleanly across the
  parameter and isn't overfitting to the training grid.
- **Latent → parameter R²**: linear regression gets near zero for most
  models, but k-nearest-neighbor regression gets 0.78–0.99. That gap is
  itself the finding — the latent organizes the parameter on a *curved*
  one-dimensional manifold inside the 2-D code. A hyperplane misses it,
  a local method follows it.
- **Silhouette score** against fixed-point vs. limit-cycle behavioral
  labels: only meaningful for FHN with I, because LV always sits in
  limit-cycle regime. There it confirms there really are two clusters.

**One honest finding worth showing**: the FitzHugh–Nagumo a-sweep gets
kNN R² ≈ 0.02 — the trained VAE didn't organize that one cleanly. We
flag this as future work; the most likely cause is that the sweep range
keeps the system in one regime, so the trajectories aren't different
enough for the encoder to spread them apart.

Transition: "Which leads us to lessons learned…"
