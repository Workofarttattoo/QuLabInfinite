# Calibration Scripts

Place executable calibration routines here. Each benchmark YAML references a
script within this directory (e.g., `mech_304ss_tension_calib.py`). Scripts
should:
1. Load the canonical dataset.
2. Perform Bayesian/optimization calibration of model parameters.
3. Emit acceptance metrics (MAE, coverage, etc.).
4. Write plots/figures into `../reports/` and structured results back to
   `../data/canonical/` if parameters change.

The current repository only includes placeholder files—populate them with real
calibration code once authoritative datasets are available.
