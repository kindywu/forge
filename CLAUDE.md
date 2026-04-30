# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

电石炉电极操作智能决策系统 — a closed-loop AI control system for a calcium carbide furnace. Uses a two-stage LightGBM causal model + SHAP explainability + safety guardrail to recommend electrode depth adjustments every 30 seconds, aiming to keep power factor in [0.92, 0.93].

## Run / develop commands

```bash
# Activate the virtual environment
.venv\Scripts\activate   

# Install dependencies
uv pip install -r requirements.txt

# Run the full demo (data generation → training → SHAP → simulation)
uv run main.py
```

There are no tests yet in this repository.

## Architecture: three-layer causal chain

This is the central design constraint — do NOT collapse these layers into a single model:

```
驱动层 (drive)      →  中间状态 (intermediate)    →  结果指标 (result)
electrode depth,      current, impedance,           power_factor,
c_cao_ratio, lime,    imbalance, temp,              energy_per_ton
coke_fixed_carbon     furnace_pressure
```

- **Model 1** (`model1`): drive features → 7 intermediate state targets (one LightGBM per target)
- **Model 2** (`model2`): intermediate state features (with historical lags/rolling stats) → power_factor + energy_per_ton (one LightGBM per target)

Both models store their per-target regressors in `Dict[str, lgb.LGBMRegressor]`.

## Key files

| File | Role |
|------|------|
| `main.py` | Entry point: generates data, trains models, runs SHAP, simulates 5 decision scenarios |
| `data/simulator.py` | Generates synthetic furnace data from first principles (physics-inspired) |
| `models/two_stage_model.py` | `TwoStageModel` class: train, save, load, predict_pipeline, SHAP importance |
| `utils/feature_engineering.py` | Builds X1/Y1 and X2/Y2 with lag features, rolling stats, and time features for both training and inference |
| `engine/decision_engine.py` | `DecisionEngine`: the 5-step closed loop (compare → simulate → score → safety-check → execute) |
| `engine/safety_guard.py` | `SafetyGuard`: hard-constraint checker independent of the AI model |

## Critical design constraints

### Time-series split (not random shuffle)

Training uses sequential split (`iloc[:split_idx]` / `iloc[split_idx:]`), never `train_test_split(shuffle=True)`. Shuffling would leak future information into the training set.

### Feature engineering for inference

There are two paths for constructing features at inference time:

- **`build_inference_features()`** (in `utils/feature_engineering.py`): The correct path. Appends the candidate row to history, re-runs the full feature pipeline (`build_features_for_model1/2`), and extracts the last row. This ensures lag/rolling features are computed correctly from the combined series.

- **`_build_simple_drive_features()`** (in `decision_engine.py`): A simplified path that fills missing features from history with `iloc[-1]`. This is a fallback and does NOT produce accurate lag features. Only the 3 electrode depth columns vary between candidates; all 100+ other features are static. Use `build_inference_features` instead.

When constructing a candidate row for inference, always:
1. Set a valid `timestamp` (e.g., `last_ts + timedelta(hours=1)`) or the row gets dropped by `dropna()` in the feature pipeline
2. Fill ALL `DRIVE_COLS` — missing columns cause NaN in the feature matrix and the row is dropped

### Safety guard is independent of the AI model

`safety_guard.py` does not import or depend on any model code. It checks hard physical/electrical constraints (current, depth range, imbalance, step size, pressure). AI recommendations that fail any check are discarded — the guard has final veto power.

### 5-step decision loop

Each `engine.step()` call runs:
1. Emergency check (PF < 0.80 → immediate human handoff)
2. Compare current PF vs [0.92, 0.93] target
3. Simulate candidates (±1/2/4 cm on each phase, ≤ 18 candidates)
4. Score: PF proximity (50%) + energy efficiency (30%) + adjustment size (20%)
5. Safety check each candidate, return best passing one or escalate

### LightGBM hyperparameters

Tuned for industrial PC deployment (small trees, fast training, low memory):
`num_leaves=31`, `learning_rate=0.05`, `n_estimators=500`, `early_stopping_rounds=50`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=5`.
