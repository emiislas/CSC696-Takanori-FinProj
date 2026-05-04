Reorganization (all moves are local; nothing staged in git):                  - src/ — all .py files                                                      
  - data/ — predator_prey_data.pt                                             
  - models/ — vae_lotkavolt.pt, decoder.onnx                                  
  - figures/ — vae_loss_curve.png
  - web/ — index.html, data_for_d3.json
  - docs/ — README, change_log, improvements, schema, originalProposal        

  Path updates — every script now computes ROOT =
  Path(__file__).resolve().parent.parent and writes/reads to the right folder 
  regardless of cwd. Updates touch solve_lotka_volterra.py,
  solve_fitzhughnagumo.py, train_vae_lotkavolt.py, plot_latent_lotkvolt.py,   
  export_for_d3.py. web/index.html now loads ../models/decoder.onnx.
  run_pipeline.sh and docs/README.md updated to match.

  New viz — src/plot_param_features.py: sweeps α over 50 values, integrates   
  Lotka-Volterra for each, computes prey amplitude (max−min)/2 and phase lag  
  predator−prey via cross-correlation (in radians of one period), and pops two
   interactive matplotlib panels via plt.show() — no PNG saved. Smoke-tested  
  end to end.

  Verified export_for_d3.py runs and writes to models/decoder.onnx and        
  web/data_for_d3.json. Git status shows deletions/untracked but nothing      
  staged, so you can review and commit when ready.
● Done.

● Done.

  Reorganization (all moves are local; nothing staged in git):
  - src/ — all .py files
  - data/ — predator_prey_data.pt
  - models/ — vae_lotkavolt.pt, decoder.onnx
  - figures/ — vae_loss_curve.png
  - web/ — index.html, data_for_d3.json
  - docs/ — README, change_log, improvements, schema, originalProposal

  Path updates — every script now computes ROOT = Path(__file__).resolve().parent.parent and writes/reads to the right folder regardless of cwd. Updates touch
  solve_lotka_volterra.py, solve_fitzhughnagumo.py, train_vae_lotkavolt.py, plot_latent_lotkvolt.py, export_for_d3.py. web/index.html now loads ../models/decoder.onnx.       
  run_pipeline.sh and docs/README.md updated to match.

  New viz — src/plot_param_features.py: sweeps α over 50 values, integrates Lotka-Volterra for each, computes prey amplitude (max−min)/2 and phase lag predator−prey via      
  cross-correlation (in radians of one period), and pops two interactive matplotlib panels via plt.show() — no PNG saved. Smoke-tested end to end.

  Verified export_for_d3.py runs and writes to models/decoder.onnx and web/data_for_d3.json. Git status shows deletions/untracked but nothing staged, so you can review and   
  commit when ready.