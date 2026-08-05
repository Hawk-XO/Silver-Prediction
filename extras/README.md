# extras/

Dev/research/demo files that aren't part of the running Streamlit app.
Nothing in `ui/`, `data/`, `features/`, `models/`, `backtest/`, `signals/`,
`broker/`, or `config/` imports anything in here -- moving this folder out
of the way does NOT affect `streamlit run ui/app.py`.

What's in here and why:

- `predictor_ui/` -- standalone React/JSX component + exported JSON. Not
  used by the Streamlit app at all; only `export_predictor_data.py` (also
  here) fed it.
- `research/` -- the feature-subset search harness used only by
  `run_feature_search.py` / `run_multi_window_check.py` below.
- `run_phase5_demo.py`, `run_phase6_demo.py`, `run_phase7_demo.py` --
  one-shot smoke-test scripts written while building each phase
  (synthetic data end-to-end through that phase's code).
- `run_feature_search.py`, `run_multi_window_check.py` -- CLI tools for
  searching/validating feature subsets.
- `export_predictor_data.py` -- generated the JSON for `predictor_ui/`.
- `ui_progress_demo.py` (was `ui/progress_demo.py`) -- standalone Streamlit
  demo script, not part of the multipage app/navigation.
- `ui_walk_forward.py` (was `ui/walk_forward.py`) -- not imported anywhere;
  leftover file, unrelated to the real `backtest/walk_forward.py` the app
  actually uses.

## If you ever want to run one of these again

They all import from the main project's top-level packages
(`data.*`, `features.*`, `backtest.*`, `signals.*`, `broker.*`, and for the
two `research`-dependent scripts, `research.*`). That only resolves if the
script is run from the project root, not from inside `extras/`. Easiest
fix: copy the specific script (and `research/`, if it's one of the two that
need it) back into the project root temporarily, then run it as the
docstring at the top of the file says, e.g.:

```
python run_phase7_demo.py
```

I didn't rewire the imports to work in-place here, since I can't run your
actual environment/dependencies in this sandbox to verify a path change
doesn't quietly break something -- safer to leave them untouched.
