# Two-Time-Scale Transfer Learning (TTSTL) Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18304963.svg)](https://doi.org/10.5281/zenodo.18304963)

Implementation of the **Two-Time-Scale Transfer Learning (TTSTL)** framework for micro-return forecasting in high-frequency limit order book (LOB) markets. This repository contains the code for the paper:

> **Two-Time-Scale Transfer Learning for Market-by-Order Micro-Return Forecasting**
> *Juan Diego Mejia Becerra, Gareth W. Peters, Ruimeng Hu*
> University of California, Santa Barbara

## 📌 Abstract
Deep Learning models face a critical "simulation-to-reality" gap when applied to high frequency financial data. Deep learning models often fail to capture the non-stationary dynamics of live markets, requiring computationally expensive recalibrations. This work introduces the Two-Time-Scale Transfer Learning (TTSTL) framework, a hybrid model architecture that combines the flexibility of deep learning models with the parsimony and calibration ease of time series regression models. The TTSTL architecture mimics the hippocampus-neocortex interaction in the human brain by coupling a high-capacity deep learning "backbone" with a lightweight, adaptive time series regression "adapter." The backbone, utilizing a CNN-LSTM network, extracts complex, non-linear feature interactions from event-driven limit order book data, while the adapter—implemented as a Hybrid ARMAX-GARCH-DeepLOB model—integrates these features into a statistical framework for micro-return forecasting. Experiments are conducted on synthetic and Intel (INTC) market-by-order data. Results demonstrate that the TTSTL framework significantly outperforms standalone deep learning baselines in terms of accuracy (RelMAE), while enabling volatility modeling and formal probabilistic forecasting.

## ⚠️ Data Availability & Reproduction
This research utilizes two types of data. Please follow the instructions below to reproduce the results.

### 1. Synthetic Data (Fully Reproducible)
We provide the exact synthetic datasets used in the paper's experiments to ensure reproducibility without re-running the generation simulations.
* **Download:** [Link to Zenodo Dataset (TBD)]
* **Generation Source:** The logic for generating this data is available in the auxiliary repository: [`evolving-microstructural-toolkit-data`](https://github.com/judmejiabe/evolving-microstructural-toolkit-data).
* **Usage:** Extract the downloaded `.parquet` files into `data/synthetic_data/`.

### 2. Real Market Data (Proprietary)
The paper analyzes high-frequency Intel (INTC) Market-by-Order data.
* **Source:** Raw data must be licensed directly from **[Databento](https://databento.com)**.
* **Preprocessing:** Raw vendor data must be normalized using the pipeline in our [MeatPy](https://github.com/judmejiabe/meatpy) repository.
* **Configuration:**
    1.  Copy `credentials.example.json` to `credentials.json`.
    2.  Fill in your Google Cloud Storage details or set the `bucket_name` to your local data path if modifying the loader.

## 🛠️ Installation

### Option A: Docker (Recommended)
We provide a containerized environment with exact Python and R versions to prevent dependency issues.

```bash
# Build the image
docker build -t ttstl .

# Run the synthetic study reproduction script
# Mounts the current directory to /app to save outputs locally
docker run --rm -v $(pwd):/app ttstl python scripts/synthetic_study_experiment.py```

### Option B: Local Install
Requires **Python 3.9+** and **R 4.3+**.

1.  **Create Environment:**
    ```bash
    conda env create -f environment.yml
    conda activate ttstl_dev
    ```
2.  **Install Package:**
    ```bash
    pip install -e .
    ```

## 🚀 Quick Start

Initialize the Orchestrator to run a training cycle with a CNN-LSTM backbone and GARCH adapter.

```python
import torch
from ttstl.orchestrator import Orchestrator, OrchestratorConfig
from ttstl.backbone import Backbone
from ttstl.architectures.cnn_lstm import CNNLSTM

# 1. Configuration
config = OrchestratorConfig(
    ticker='SIM',
    bdelta=5000000.0,  # Backbone update interval (ns)
    adelta=100000.0,   # Adapter update interval (ns)
    sdelta=100.0       # Sampling interval (ns)
)

# 2. Initialize Orchestrator
# (Assuming you have preloaded dataframes ed_df and ptd_df)
orchestrator = Orchestrator(run_dir="outputs/test_run", config=config)

# 3. Run Global Update (Backbone Training)
orchestrator.global_update(ed_df, ptd_df)

# 4. Run Local Adaptation (Adapter Fitting)
# The adapter will now use features extracted by the trained backbone
orchestrator.local_adapt(ed_df_next, ptd_df_next)
```

## 📂 Repository Structure
* `src/ttstl/`: Core package source code.
    * `backbone.py`: Deep learning feature extractor management.
    * `adapter.py`: R-based statistical transfer learning module (rpy2 interface).
    * `orchestrator.py`: Manages the two-time-scale interaction.
* `scripts/`: Reproduction scripts for paper experiments.
    * `synthetic_study_experiment.py`: Main script for synthetic data results.
    * `market_data_experiments.py`: Script for real market data analysis.

## 🔗 Citation
If you use this code, please cite our paper:

```bibtex
@article{mejia2026ttstl,
  title={Two-Time-Scale Transfer Learning for Market-by-Order Micro-Return Forecasting},
  author={Mejia Becerra, Juan Diego and Peters, Gareth W. and Hu, Ruimeng},
  year={2026},
  institution={University of California, Santa Barbara}
}
```