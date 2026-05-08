# Final Report — Structure & Content Plan

Target: a short workshop-style paper presenting a visual analytics system for
exploring parametrized ODE solutions through a learned 2-D VAE latent space.
Section weights below mirror the rubric (`rubric.md`) so length/effort can be
budgeted accordingly.

---

## 1. Title, Authors, Abstract

- One-paragraph abstract (≤200 words) covering: problem, approach (VAE +
  coordinated views), key result (kNN R² 0.78–0.99 on 4/5 models, live
  in-browser decoding), and the analytic capability the system unlocks
  ("clicking unseen points to synthesize new trajectories").
- Keep abstract self-contained — no jargon that isn't defined.

---

## 2. Introduction & Motivation  *(rubric §b — 2 pts)*

- Frame the analyst's pain: parameter studies of ODEs traditionally require
  re-solving for every new parameter value, and the result is a stack of
  static trajectory plots that hides the *organization* of the family of
  solutions.
- State the goal: a visual analytics system where the parameter domain itself
  is a navigable canvas.
- Why VA, not just ML: the learned latent is only useful if it can be
  *interrogated* — hover, click, decode, compare. Position the contribution
  as the integration, not the model alone.
- Scope statement: 2-channel ODEs (Lotka–Volterra, FitzHugh–Nagumo), single
  varied parameter per model, 2-D latent space.
- Bulleted summary of contributions at the end of the section (3–4 bullets).

**Figures here:**
- **Fig. 1 — Teaser.** Composite screenshot of the running tool: latent
  scatter, trajectory panel, parameter-feature panel, equation panel. One
  caption that walks the eye through the four views.

---

## 3. Related Work & Positioning  *(rubric §c — 2 pts)*

This is one of the rubric items the reviewer specifically flagged. Treat it
as a positioning section, not a literature dump.

- Group prior work into three buckets and cite ≥1 paper per bucket:
  1. **Static / animated trajectory visualization for ODEs** (textbooks,
     phase-portrait tools, bifurcation diagram software like XPPAUT, MATCONT).
     Note that these show *one* projection at a time.
  2. **VAEs and latent-space visualization in scientific ML** (e.g. β-VAE for
     disentanglement, latent-space exploration in molecular / image domains).
     Note that these typically don't expose the latent to a user as an
     analysis surface.
  3. **Coordinated multiple views in visual analytics** (Roberts; brushing &
     linking literature). Note that these are usually applied to tabular or
     spatial data, not learned latent spaces of dynamical systems.
- **Explicit positioning paragraph** (this is what the reviewer asked for):
  - Existing online ODE visualizations almost universally show *only* the
    input trajectory — either statically or as an animation.
  - Our system simultaneously presents (i) the trajectory, (ii) phase angle,
    (iii) amplitude, and (iv) a 2-D learned latent embedding, all linked.
  - The latent view is *generative*: clicking empty space synthesizes a
    new trajectory, which existing tools cannot do.
  - The latent view is *unsupervised*: the parameter map is discovered, not
    drawn — bifurcation diagram tools require analytic continuation, ours
    falls out of training.
  - The framework is *system-agnostic*: a registry entry adds a new ODE/
    parameter pair without touching the frontend.

**Figures here (optional but strong):**
- **Fig. 2 — Positioning sketch.** A 2x2 table or small diagram contrasting
  prior approaches vs. this work along axes like *unsupervised, generative,
  multi-view, system-agnostic*. Even a simple table is fine.

---

## 4. Problem Formulation  *(supports §b and §d)*

Short — half to one page. Set up notation that the rest of the report will
reuse.

- ODE family $\dot{x} = f(x;\theta)$ with $\theta$ scalar in this work.
- Goal: learn $E_\phi: \mathbb{R}^{T \times 2} \to \mathbb{R}^2$ and
  $D_\psi: \mathbb{R}^2 \to \mathbb{R}^{T \times 2}$ such that the 2-D code
  $z = E_\phi(x_{\text{window}})$ varies monotonically with $\theta$.
- Analysis tasks the system supports — phrase as user-facing tasks T1–T4:
  - **T1.** Identify which latent direction encodes the swept parameter.
  - **T2.** Compare trajectory shape across parameter values without
    re-solving.
  - **T3.** Synthesize an unseen-parameter trajectory by clicking empty
    latent space.
  - **T4.** Detect qualitative regime changes (e.g. Hopf bifurcation) as
    geometric features in the latent space.

---

## 5. Technical Approach  *(rubric §d — 5.5 pts, the highest-weighted section)*

### 5.1 Data Generation

- Two ODE families, five (system, parameter) pairs (cite the table from the
  schema).
- Sequential windowing strategy and why (uniform latent coverage, mentioned
  in the schema). Window size 350, 40 windows/trajectory, 800 examples/model.
- Per-feature standardization; mean/std saved for in-browser denormalization.

### 5.2 VAE Architecture & Training

- 3-layer MLP encoder → (μ, log σ²) → 2-D z → 3-layer MLP decoder, BatchNorm
  + Tanh.
- ELBO with β=0.01 and 50-epoch linear KL warm-up (justify both: low β to
  prioritize reconstruction; warm-up to avoid posterior collapse).
- 500 epochs, Adam, ReduceLROnPlateau.
- Mention that the same hyperparameters work across all five models — this
  is part of the "system-agnostic" claim.

### 5.3 Visualization & Interaction Design

- Three coordinated views, with a justification rooted in the analysis
  tasks T1–T4:
  - **Latent scatter** — answers T1, T4.
  - **Trajectory plot** — answers T2 (and T3 when paired with the decoder).
  - **Parameter-feature plots** (parameter vs. amplitude, parameter vs.
    phase angle) — give the latent a *physical* axis to compare against
    (answers T1 quantitatively in the user's head).
- Linked highlighting across all three views.
- **In-browser ONNX decoder** for click-to-synthesize. Justify why this
  matters: zero server round-trip means the crosshair feels real-time, which
  changes the interaction from "query" to "exploration."
- Equation panel with gold-highlighted varied parameter — design rationale
  is grounding for users who don't have all the symbols memorized.
- Five-button model switcher; lazy-load and cache strategy.

### 5.4 Use of AI Tools (required disclosure)

- Document any LLM use (e.g. for boilerplate code, doc generation, this
  outline). State that all generated code was tested and that quantitative
  claims were verified against the script in `evaluation/evaluate.py`.

**Figures here:**
- **Fig. 3 — System architecture diagram.** Boxes for: ODE solver →
  windowing → VAE training → ONNX export → frontend (latent / trajectory /
  feature views), with arrows.
- **Fig. 4 — VAE architecture diagram.** Encoder, reparameterization,
  decoder, including layer sizes.
- **Fig. 5 — Annotated UI screenshot.** Same as Fig. 1 but with callouts
  pointing at each view, the equation panel, the switcher, and the
  click-to-decode crosshair.

---

## 6. Implementation  *(rubric §e — 3.5 pts)*

- Repository layout (mirrors the README table).
- Backend: PyTorch for training, `scipy.solve_ivp` for data, ONNX export via
  `torch.onnx.export` (opset 14, dynamic batch axis).
- Frontend: vanilla JS + D3 for views, `onnxruntime-web` for decoding,
  MathJax for equations. No build step.
- Pipeline reproducibility: `run_pipeline.sh` (and `_mac.sh`) chain
  `solve_odes.py → train_vae.py → export_for_d3.py` for all five models.
- Adding a new ODE = one registry entry in each of the three pipeline
  scripts; the frontend picks it up automatically. Highlight this — it's the
  generalization story the reviewer asked about.

**Figures here:**
- **Fig. 6 — Pipeline diagram.** Flow from `solve_odes.py` to running web
  app, marking which artifacts (`.pt`, `.onnx`, `.json`) cross which
  boundary.

---

## 7. Evaluation  *(rubric §f — 4.5 pts; reviewer flagged depth here)*

The reviewer explicitly said this section was thin in the presentation. Add
material in three places: case studies, user feedback, quantitative metrics.

### 7.1 Case Studies (qualitative)

For each of the five models, one short paragraph + one figure:
- **Lotka–Volterra α** — clean 1-D arc; latent tracks frequency.
- **Lotka–Volterra γ** — structurally identical (ω = √(αγ)); shown to
  confirm the framework picks up the same geometry from a different
  parameter.
- **Lotka–Volterra β** — independent of δ; latent encodes both frequency
  and equilibrium shift; waveform shape changes visibly along the arc.
- **FitzHugh–Nagumo a** — *honest negative result*: kNN R² ≈ 0.02. Discuss
  why (sweep range likely doesn't cross enough qualitative regimes).
- **FitzHugh–Nagumo I** — latent splits into two visible blobs across the
  Hopf bifurcation; this is the strongest "VA insight" finding because the
  bifurcation appears as a *spatial* feature without supervision.

### 7.2 User Feedback

This is the part the reviewer specifically asked for.

- A short paragraph summarizing what the audience said during the live demo
  (e.g. switcher is intuitive, the equation-panel grounding helps,
  click-to-decode was the most engaging interaction).
- Then **embed a screenshot of the actual feedback** (Fig. 8 below) and
  walk through the specific comments — quote at least 1–2 directly. Tie
  each comment back to a design decision in §5.3.

### 7.3 Quantitative Evaluation

Lift the table from `evaluation/results.md` and walk through:
- **Reconstruction MSE** vs. **Generalization MSE** — emphasize they're
  approximately equal across all five models, which means the VAE
  interpolates between training parameter values cleanly (a generalization
  claim, not just a fit claim).
- **R² linear vs. R² kNN** — the gap is itself the finding: the latent
  organizes the parameter on a *curved* 1-D manifold inside the 2-D code.
- **Silhouette** — meaningful only for FHN-I (only model crossing a
  bifurcation in its training range). Confirms the qualitative two-cluster
  finding from §7.1.
- Caveat the held-out split (unseeded) and explain the generalization MSE
  as the real test.

### 7.4 Validation of AI-Assisted Content

- If LLMs were used to draft any code (e.g. evaluation harness), state that
  numbers in the table came from `evaluate.py` runs you executed and
  spot-checked, not from generated text.

**Figures here:**
- **Fig. 7 — Five-panel latent space gallery.** One scatter per model, same
  layout, same colorbar style. Lets the reader see at a glance how geometry
  differs.
- **Fig. 8 — User feedback screenshot.** The actual screenshot you have of
  classmate feedback. Highlight 1–2 specific quotes.
- **Fig. 9 — `metrics_summary.png`.** Already produced by `evaluate.py`;
  just embed and reference.
- **Fig. 10 — Loss curves.** A grid of the five `vae_loss_curve_*.png`
  files for transparency on training.
- **Fig. 11 — Click-to-decode example.** Two-panel: (left) latent space
  with crosshair on a point *between* training parameter values, (right)
  the synthesized trajectory. Caption: "no simulation in the training set
  has these latent coordinates."

---

## 8. Discussion, Limitations, Lessons Learned  *(rubric §g — 2 pts)*

Be candid; the rubric explicitly rewards critical reflection.

- **What worked.** Coordinated views; in-browser decoding; registry-driven
  generalization; latent space is interpretable for 4/5 models.
- **What didn't.** FHN-a R² ≈ 0.02. Hypothesize causes (sweep range
  doesn't cross a regime change; possibly latent dim too small to encode
  multiple slow features). State this is future work.
- **Scaling to high-dimensional equations** — reviewer asked for this:
  - The VAE is dimension-agnostic on the input side: replacing the 2-channel
    window with an N-channel one changes only the input layer width.
  - The latent dim does *not* need to scale with state dim — it should
    scale with the number of *swept parameters*, since the manifold
    dimensionality is determined by the parameter sweep, not the state.
  - The generalized architecture (one registry entry per system) makes this
    extension a concrete next step, not a redesign.
  - Caveat: visualizing latent dim > 3 requires a different UI (parallel
    coordinates, projections like UMAP-on-latents). Discuss explicitly.
- **Limitations.** Single-parameter sweeps only in this iteration; 2-D
  latent only; ~3 minutes/model train cost negligible but doesn't reflect
  cost at higher state dimension.
- **Lessons learned.** Pick 1–2 surprises from the project — e.g. that the
  *spatial* shape of the latent encodes regime change without any explicit
  bifurcation analysis; that β annealing mattered more than expected; that
  in-browser ONNX inference made the interaction qualitatively different.

---

## 9. Future Work  *(rubric §h — 1 pt)*

Three concrete directions, each tied to current work:

1. **Multi-parameter sweeps.** Vary 2+ parameters and bump latent dim
   accordingly; investigate disentanglement (β-VAE, FactorVAE).
2. **Higher-dimensional ODEs / PDEs.** Test on a 3+ channel system (e.g.
   Hodgkin–Huxley, Lorenz). Per the discussion above, this is mostly an
   input-layer change.
3. **Bifurcation-aware design.** When kNN R² is low (FHN-a case), surface
   that automatically and warn the user, or suggest expanding the parameter
   range. Connects directly to the FHN-a negative result.

---

## 10. References

- Kingma & Welling — original VAE.
- Higgins et al. — β-VAE / disentanglement.
- FitzHugh / Nagumo papers (or a textbook).
- Lotka, Volterra (or a textbook).
- Roberts (or similar) for coordinated-views VA.
- XPPAUT / MATCONT for bifurcation tooling positioning.
- Anything cited in §3.

Aim for ≥6 references to clear the rubric §c bar of "multiple relevant
papers cited."

---

## Appendix (optional)

- Full hyperparameter table (already in `docs/schema.txt`).
- Per-model display-parameter subsets and color ramps.
- Reproduction commands.

---

## Master figure list (consolidated)

| #  | Figure                             | Source / how to make it                                  |
|----|------------------------------------|----------------------------------------------------------|
| 1  | Teaser composite screenshot        | Browser screenshot of the live tool, four views visible  |
| 2  | Positioning table/sketch           | Hand-drawn or simple table                               |
| 3  | System architecture diagram        | Draw fresh (boxes + arrows)                              |
| 4  | VAE architecture diagram           | Draw fresh from `docs/schema.txt` §9                     |
| 5  | Annotated UI screenshot            | Screenshot + callouts                                    |
| 6  | Pipeline diagram                   | Draw fresh from §13 of `docs/schema.txt`                 |
| 7  | Five-panel latent gallery          | Render from each model's exported JSON / `export_for_d3` |
| 8  | User feedback screenshot           | Already have — embed                                     |
| 9  | Quantitative metrics bar chart     | `evaluation/metrics_summary.png` (already exists)        |
| 10 | Training loss curves grid          | Combine the five `figures/vae_loss_curve_*.png`          |
| 11 | Click-to-decode example            | Screenshot of crosshair in empty latent space + decoded trajectory |

---

## Suggested length / weight allocation

| Section                            | Pts  | Approx. length    |
|------------------------------------|------|-------------------|
| Intro & Motivation                 | 2    | 0.5–1 page        |
| Related Work & Positioning         | 2    | 0.75 page         |
| Problem Formulation                | —    | 0.25–0.5 page     |
| Technical Approach                 | 5.5  | 2–2.5 pages       |
| Implementation                     | 3.5  | 1 page            |
| Evaluation                         | 4.5  | 2 pages           |
| Discussion / Limitations           | 2    | 0.75 page         |
| Future Work                        | 1    | 0.25 page         |
| Org / Writing Quality              | 2    | (cross-cutting)   |
| **Total**                          | 22.5 | **~8–9 pages**    |
