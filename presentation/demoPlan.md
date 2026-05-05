# Demo Plan

- start with math equation panel and selecting an equation
- explain latent space, amplitude, phase
- show linked views
- show interpolation
- explain how each equation's visualization lines up correctly




## Pre-demo checklist (do this BEFORE you start screen-sharing)

- [ ] `python -m http.server 8000` running from the project root.
- [ ] Browser tab open at `http://localhost:8000/web/index.html`.
- [ ] Cache cleared once so the ONNX files load fresh (Ctrl+Shift+R).


## 1. Open with the latent space view  *(~45 s)*

1. Point at the latent scatter on the left.
2. Say: *"Every dot is a windowed simulation. Color is the value of α
   the simulation was run at. We never told the network what α was —
   the encoder put them on this curve by itself."*
3. Drag the mouse along the arc and call out the color gradient: low α
   on one end, high α on the other.
4. Mention: 2-D latent space, 800 windows, only ~3 minutes to train.

**Insight to land**: the parameter map is *emergent*, not hand-crafted.

---

## 2. Click a training sample → linked highlighting  *(~45 s)*

1. Click a dot in the middle of the arc.
2. Three things light up simultaneously:
   - The dot itself in latent space.
   - The matching trajectory appears in the trajectory plot (top right).
   - The matching point lights up in *both* parameter-feature plots
     (amplitude and phase, bottom right).
3. Say: *"All three views are coordinated. This dot is one specific
   simulation, and we're now seeing every projection of it at once."*
4. Click two or three dots to show how amplitude and phase vary across
   the latent.

**Insight to land**: the same datum has many projections, and the user
can pivot between them with one click.

---

## 3. Click empty latent space → decode a novel trajectory  *(~75 s)*

1. Click somewhere in empty latent space — *between* the colored dots.
2. Point out: a yellow trajectory appears. *"This was synthesized by the
   decoder right now, in your browser. There's no simulation in the
   training set with these latent coordinates."*
3. Drag the crosshair around. The trajectory updates live as you move.
4. Move toward the start of the arc, then to the end — show how the
   waveform period and amplitude change continuously.
5. Mention briefly: the decoder runs as ONNX in the browser via
   `onnxruntime-web`, no server, no Python — the response is real-time
   because there's no round-trip.

**Insight to land**: the latent space isn't just a *map* of existing
simulations, it's a *generator* of new ones. This is the part the user
study found most engaging.

---

## 4. Switch models — show the parameter changes the geometry  *(~75 s)*

1. Click **Lotka–Volterra (β)** in the switcher.
2. The whole UI updates: scatter, trajectories, equation panel (note
   the highlighted β in gold), color ramp.
3. Click a low-β dot, then a high-β dot.
4. Call out: *"Look at the trajectory shape. Low β is spiky and
   explosive. High β is smooth and small-amplitude. This is qualitative
   information you can only see by clicking around — it's the kind of
   thing that's hard to surface from a static plot of a 5-parameter
   sweep."*
5. Switch to **FitzHugh–Nagumo (I)**. Show that the latent space is
   visibly *split* into two clouds — fixed-point trajectories on one
   side, limit-cycle on the other.
6. Click one of each. Show the dramatically different trajectory
   shapes.
7. Say: *"This is the Hopf bifurcation appearing as a geometric feature
   in the latent space, with no supervision."*

**Insight to land**: the framework isn't tied to one ODE; the same UI
surfaces *different kinds* of structure depending on the system.

---

## 5. Equation panel and varied-parameter highlight  *(~30 s)*

1. Point at the equation panel between the switcher and the plots.
2. Switch one more time and call out the gold-highlighted parameter
   updating in the equations.
3. Say: *"This is a small detail but it grounds the user in what's
   actually changing — particularly important for an audience that
   hasn't memorized which letter is which in Lotka–Volterra."*

**Insight to land**: the visualization keeps the math visible.

---

## 6. (If time) Pop the evaluation results table  *(~30 s)*

1. Open `evaluation/results.md` in a second tab.
2. Point at the kNN R² column: 0.78–0.99 for four of the five models,
   showing the latent really does encode the parameter.
3. Point at the FitzHugh-Nagumo (a) row showing 0.02 — *"this is the
   one case where it didn't work, which is itself a finding."*

**Insight to land**: the qualitative story is backed by numbers.

---

## Wrap  *(~30 s)*

> "So to recap what you saw: an unsupervised 2-D parameter map,
> coordinated trajectory and feature views, a live decoder that
> synthesizes new trajectories from anywhere in the latent space, and a
> system design that scales to arbitrary 2-channel ODEs by adding one
> registry entry."

---

## Fallbacks if something breaks

- **ONNX fails to load**: refresh the page; if still broken, switch
  back to Lotka–Volterra α (decoder.onnx) which is the most-tested
  path.
- **Crosshair decoder produces noise**: that means the latent point is
  far outside the training distribution — move closer to the colored
  arc and explain that as an out-of-distribution effect.
- **Page doesn't load at all**: roll the backup screen recording.
- **Asked an unexpected question mid-demo**: pause the demo, answer
  from the speaker notes, then return to the next checkpoint above.
