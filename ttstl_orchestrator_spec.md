# TTSTL Orchestrator — Specification

**Location:** `src/ttstl/orchestrator.py`  
**Audience:** Senior Quant / ML Engineer  
**Scope:** Instantiate and coordinate **Backbone**, **Interface**, and **Adapter** on a *single backbone bar* at a time. Persist artifacts in a deterministic `runs/` layout. No data provider is injected; both public methods accept in-memory `pandas.DataFrame`s.

---

## 1) Responsibilities & Boundaries

### Owns
- Construction of **Backbone**, **Interface**, **Adapter** (internally; not injected). The constructor should create a backbone → interface → adapter stack per backbone time bar as determined by the trading session intervals.
- Creation of a **run directory** named with a reproducible slug (see §4).
- Two public actions per backbone bar:
  1. `global_update(event_df, phys_df)`: full update of backbone → interface → adapter. The class names should match the script ones. Each class has its own updating methods.
  2. `local_adapt(event_df, phys_df)`: the learned parameters in the backbone and interface are transferred. The adapter module is updated
- Writing all artifacts in a stable folder hierarchy (data + figures).
- Seeding RNGs (Python/NumPy/PyTorch; optionally R) and writing `config.yaml` and `manifest.json`.

### Does **not** own
- Data access or slicing. Both methods accept event driven and physical time driven trading session data in the form of DataFrames. A column indicator of the backbone and adapter time bar is automatically created by the constructor class.
  - `event_df`: event-driven trading session data
  - `phys_df`: physical-time trading session data

---

## 2) Public API

```python
class Orchestrator:
    def __init__(self, runs_root: pathlib.Path, cfg: OrchestratorConfig) -> None: ...
    def global_update(self, event_df: pd.DataFrame, phys_df: pd.DataFrame) -> dict: ...
    def local_adapt(self, event_df: pd.DataFrame, phys_df: pd.DataFrame) -> dict: ...
```
- `__init__`
  - Seeds RNGs.
  - Builds run slug and creates the top-level run directory.
  - Writes `config.yaml` and initial `manifest.json`.
  - Instantiates `Backbone`, `Interface`, and `Adapter` using `cfg` (including ENet/adaLASSO grids).
  - Initializes internal backbone counter `self._bb_idx = -1`.

- `global_update(...)`
  - Increments `self._bb_idx` and creates `bb_{idx:03d}/global_update/...`. Receives event and physical time driven data for a full trading session. Creates two columns indicating the backbone and adapter time bars in the datasets if not available.
  - Trains backbone (CNN-LSTM), runs interface global selection (Elastic Net + adaptive LASSO), then adapter path; persists artifacts under **backbone/**, **interface/**, and **adapter** (see §6).
  - Returns dict with key metrics, selected variables, and artifact paths.

- `local_adapt(...)`
  - Uses current `bb_{idx:03d}/local_adapt/...` subtree. Receives event and physical time driven data for a full trading session. Creates two columns indicating the backbone and adapter time bars in the datasets if not available.
  - Runs local_adaptation from the interface and update_adapter for the adapter; mirrors the global directory layout (see §6).
  - Returns dict with OOS metrics, adapter bar specific coeffs, and break decisions.

> A `run_session(...)` method may be added later to loop over multiple backbone bars; **not required now**.

---

## 3) Configuration

```python
from dataclasses import dataclass
from typing import List, Optional, Union

@dataclass
class OrchestratorConfig:
    # Identity & timing
    ticker: str                      # e.g. "MSFT"
    bdelta: float                    # Defines backbone time bar length
    adelta: float                    # Defines adapter time bar length
    sdelta: float                    # Defines sampling time bar length
    seed: int                        # RNG seed
    session_start: float             # start of the session in the same units as the recorded timestamp
    session_finish: float            # end of the session in the same units as the recorded timestamp

    # Backbone / Interface / Adapter knobs
    quantile: float = 0.005
    parsimonious_mode: bool = False
    batch_size: int = 32
    epochs: int = 12
    selection_threshold: float = 0.15
    tolerance_threshold: float = 1e-6
    ols_significance_level: float = 0.05

    # Missing train/validation split

    # Hyperparameter grids (explicitly included)
    alpha_enet_grid: List[float] = None
    lambda_enet_grid: List[float] = None
    tau_adalasso_grid: List[float] = None
    lambda_adalasso_grid: List[float] = None

    # Optional run-id override
    run_id: Optional[str] = None
```

---

## 4) Run Name (Slug) & Top-Level Directory

**Format**  
`[Date]_[TICKER]_bdelta[ΔB]_adelta[ΔA]_sdelta[ΔS]_seed[seed]_id[ID]`

**Example**  
`2025-08-14_MSFT_bdelta100000000_adelta100000_sdelta100_seed42_id8f1a2c`

**Rules**
- `[Date] = YYYY-MM-DD` (consistent local/UTC per project policy).
- If `cfg.run_id is None`, generate a short random id (e.g., `secrets.token_hex(3)`).

**At `__init__` time**, write into the run root:
- `config.yaml` (serialization of `OrchestratorConfig`)
- `manifest.json` (initial provenance: code/env snapshot, session window, seed)

---

## 5) Reproducibility & Provenance

- Seed Python, NumPy, PyTorch (and CUDA if available); optionally seed R via rpy2.
- **manifest.json** should include:
  - `run_slug`, `ticker`, `bdelta`, `adelta`, `sdelta`, `seed`, `session_start`, `session_finish`
  - code provenance (git commit/branch/dirty)
  - environment versions (Python, Torch, Pandas, NumPy, rpy2, R, glmnet, etc.)
  - artifact paths added incrementally by `global_update` / `local_adapt`
- **config.yaml** stores the full `OrchestratorConfig`.
- Use atomic writes for large artifacts: write `.tmp` then `os.replace()`.

---

## 6) Runs Directory Structure (Per Backbone Bar)

Local-adapt mirrors the same subdirectories as global-update (structure parity). **Adapter visuals are stored under `interface/`** per the specification.

```
runs/
  [RUN_SLUG]/
    manifest.json
    config.yaml

    bb_000/
      global_update/
        backbone/
          architecture/
            train_log.parquet             # Train and validation MSE per epoch (Check)
            best.pt                       # Serialized weights and optimizer state at best validation loss (Check)
            checkpoints/
              epoch_0005.pt               # Add a checkpoint of the serialized weights and optimizer state every 5 epochs to recover from disruptions (Check: Saved a checkpoint on every epoch instead of five)

          stationarity/
            stationarity_raw.parquet      # Row-wise outcomes of ADF/PP/KPSS per variable × adapter_bar. Each row should contain variable, bar, test, and rejection (Check)
            stationarity_summary.parquet  # aggregated rates per variable (Check)

          correlation/
            corr_matrix.npz               # np.savez_compressed: {corr, vars} (Check)
            corr_heatmap.png              # Correlation matrix heatmap (Check: There are three heatmas when running in full mode. Not only one)

          extended_covariates.json        # full covariate set after pruning for stationarity (Check)
          nn_weights.npz                  # layer weights snapshot (Check, moved to the architecture subdirectory)

        interface/ (Added enet_coef_corridor.png, selected_variables.json, and variable_selection_info.json)
          corridor_ada_lasso.png        # Corridor plot of adaptive lasso using the selected hyperparameters
          hyperparams_selected.json     # chosen α_enet, λ_enet, τ, λ_ada, thresholds (Check)
          enet_hyperparams_grid_eval.parquet # grid search trace for enet hyperparameters (MSE path) (Check)
          adalasso_hyperparams_grid_eval.parquet # grid search trace for adalasso hyperparameters (BIC path) (Check)

        adapter/
          armax_garch_corridor.png        # corridor plot of adapter bar specific estimated parameters
          param_series.png            # Bar-specific and combined ARMAX-GARCH parameter time series overimposed. The x-axis corresponds to the adapter time bar
          performance_is.parquet          # in-sample RelMAE, RelMSE, and R^2.

      local_adapt/
        backbone/
          architecture/
            test_os.json                    # test MSE of the trained architecture (Pending verification)

        interface/
          corridor_ada_lasso.png          # adaptive lasso corridor plot with selected hyperparameters
          structural_breaks.parquet       # table indicating statistic and p-value of structural break test per adapter bar

        adapter/
          armax_garch_corridor.png        # corridor plot of adapter bar specific estimated parameters.
          param_series.parquet            # Bar-specific and combined ARMAX-GARCH parameter time series overimposed. The x-axis corresponds to the adapter time bar
          performance_os.parquet          # Out-of-sample RelMAE, RelMSE, and R^2.

    bb_001/
      ...
```

---

## 7) `global_update(event_df, phys_df)` — Required Steps

1. **Create backbone scope**
   - Increment `self._bb_idx` → make `bb_{idx:03d}/global_update/{backbone,interface,adapter}/`.

2. **Backbone**
   - Train CNN-LSTM on this backbone bar using internal time-based validation split
     (90/10 with 5% buffers).
   - Save:
     - `backbone/architecture/train_log.parquet` (per epoch/step metrics)
     - `backbone/architecture/best.pt` (+ checkpoints every 5 epochs at `backbone/architecture/checkpoints/`)
   - Extract features; compute diagnostics and persist:
     - Stationarity: `stationarity/stationarity_raw.parquet`, `stationarity/stationarity_summary.parquet`
     - Correlation: `correlation/corr_matrix.npz` + `correlation/corr_heatmap.png`
     - Extended covariates: `extended_covariates.json`
     - NN weights in np.savez_compressed format: `nn_weights.npz`

3. **Interface (Global Selection)**
   - Tune **Elastic Net** (`alpha_enet_grid`, `lambda_enet_grid`) via validation MSE.
   - Tune **Adaptive LASSO** (`tau_adalasso_grid`, `lambda_adalasso_grid`) via BIC.
   - Persist:
     - `corridor_ada_lasso.png` (adaLASSO corridor)
     - `hyperparams_selected.json` (final α/λ/τ/λ_ada & thresholds)
     - `enet_hyperparams_grid_eval.parquet` MSE grid search path. The dataset should contain the columns `alpha`, `lambda_enet`, and `MSE`
     - `adalasso_hyperparams_grid_eval.parquet` BIC grid search path. The dataset should contain the columns `tau`, `lambda_ada` and `BIC`

4. **Adapter**
   - Fit ARMAX–GARCH path for this backbone bar
     - `adapter/armax_garch_corridor.png` Corridor plot for the adapter bar specific estimates with significance indicators
     - `adapter/param_series.parquet` Adapter bar number vs adapter bar specific and combined ARMAX-GARCH parameters plot. 
     - `adapter/performance_is.parquet` In sample performance metrics

5. **Manifest update**
   - Append artifact paths and key metrics; return a summary dict.

---

## 8) `local_adapt(event_df, phys_df)` — Required Steps

1. **Create local scope**
   - Work under `bb_{idx:03d}/local_adapt/{backbone,interface,adapter}/`.

2. **Backbone (OS)**
   - With frozen CNN-LSTM from GU, compute **OS NN metrics** per backbone bar and save:
     - `backbone/test_os.json`

3. **Interface (Local Selection)**
   - For each backbone bar:
     - Run local adaLASSO with learned hyperparameters + Wald test; build local corridor and structural breaks table:
       - `interface/corridor_ada_lasso.png` Interface corridor plot.
       - `interface/structural_breaks.parquet` Structural break dataset. Should contain the columns `wald_statistic` and `p_value`

4. **Adapter (Local)**
   - Update ARMAX–GARCH on each adapter bar. Capture:
     - `adapter/armax_garch_corridor.png` Corridor plot for the local estimates with significance indicators
     - `adapter/param_series.parquet` Adapter bar number vs adapter bar specific and combined ARMAX-GARCH parameters plot. 
     - `adapter/performance_os.parquet` Out of sample performance metrics.

5. **Manifest update**
   - Append local artifacts and metrics; return summary dict.

---

## 9) Logging & Errors

- Use `logging.getLogger("ttstl.orchestrator")`.
- On R/rpy2 or per-bar failures in selection/adapter: log a warning; continue to next adapter bar.
- On global backbone/selection failure: raise `RuntimeError` (cannot proceed without GU artifacts).
- Prefer **atomic writes** for large artifacts.

---

## 10) Open Implementation Points

- Provide public wrappers in `Backbone` and `Interface` to avoid calling private methods (e.g., a
  `prepare_augmented_data` wrapper for `Interface._prepare_data`).
- If needed later, add `run_session(session_id)` to iterate multiple backbone bars by delegating to
  `global_update` + `local_adapt`.
- Implement goodness of fit measures. GOF corridor plot and Cross Correlation Functions on the residuals.
