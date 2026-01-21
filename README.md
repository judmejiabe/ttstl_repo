# Two-Time-Scale Transfer Learning (TTSTL) Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18304963.svg)](https://doi.org/10.5281/zenodo.18304963)

Implementation of the **Two-Time-Scale Transfer Learning (TTSTL)** framework for micro-return forecasting in high-frequency limit order book (LOB) markets. This repository contains the code for the paper:

> **Two-Time-Scale Transfer Learning for Market-by-Order Micro-Return Forecasting**
> *Juan Diego Mejia Becerra, Gareth W. Peters, Ruimeng Hu*
> University of California, Santa Barbara

## 📌 Summary
Deep Learning models face a critical "simulation-to-reality" gap when applied to high frequency financial data. Deep learning models often fail to capture the non-stationary dynamics of live markets, requiring computationally expensive recalibrations. This work introduces the Two-Time-Scale Transfer Learning (TTSTL) framework, a hybrid model architecture that combines the flexibility of deep learning models with the parsimony and calibration ease of time series regression models. The TTSTL architecture mimics the hippocampus-neocortex interaction in the human brain by coupling a high-capacity deep learning "backbone" with a lightweight, adaptive time series regression "adapter." The backbone, utilizing a CNN-LSTM network, extracts complex, non-linear feature interactions from event-driven limit order book data, while the adapter—implemented as a Hybrid ARMAX-GARCH-DeepLOB model—integrates these features into a statistical framework for micro-return forecasting. Experiments are conducted on synthetic and Intel (INTC) market-by-order data. Results demonstrate that the TTSTL framework significantly outperforms standalone deep learning baselines in terms of accuracy (RelMAE), while enabling volatility modeling and formal probabilistic forecasting. The TTSTL architecture is better understood with the diagram below. For a detailed description of each module, refer to Section 6.1 of the paper.

![TTSTL Architecture](images/TTSTL.png)

## ⚠️ Data Availability & Reproduction

**Disclaimer:** Reproducing the results presented in this paper requires access to proprietary market data and significant computational resources.

* **Hardware Specification:** All computational experiments were conducted on a remote machine equipped with an Intel Core i9-9940X CPU (14 cores, 28 threads, 3.30 GHz base clock), 125 GiB of RAM, and four NVIDIA GeForce RTX 2080 Ti GPUs (each with 11 GiB VRAM). Primary storage consisted of a 1.8 TB NVMe SSD, supplemented by a 3.5 TB HDD for data storage. The machine ran Ubuntu 20.04.6 LTS (Kernel 5.4.0-200-generic).

* **Compute Time:** The synthetic data experiments took approximately 40 hours. The real market data experiments required approximately 22 hours for CNN-LSTM pretraining and an additional 28.5 hours for the experimental runs.

### Synthetic Data (Fully Reproducible)
We provide the exact synthetic datasets used in the paper's experiments to ensure reproducibility without re-running the generation simulations.
* **Download:** [https://doi.org/10.5281/zenodo.18304963](https://doi.org/10.5281/zenodo.18304963)
* **Generation Source:** Alternatively, you can re-generate the data using the codebase in our auxiliary repository: [`evolving-microstructural-toolkit-data`](https://github.com/judmejiabe/evolving-microstructural-toolkit-data).

### Real Market Data (Proprietary)
The paper analyzes high-frequency Intel (INTC) Market-by-Order (MBO) data and VIX index data.
* **MBO Data:** Raw market-by-order data must be licensed directly from **[Databento](https://databento.com)**. As of January 21st, 2026, **Databento** offer 125 USD in free credits towards subscription plans which suffices to acquire the data required for the experiments in this paper.
* **VIX Data:** Volatility index data must be sourced from **[Massive.com](https://massive.com)**. Previously known as **[Polygon.io](https://polygon.io)**. This service requires a subscription plan.
* **Processing Instructions:** This repository only contains high-level instructions on how to download the data. For full instructions on downloading and processing this data, please refer to our dedicated data repository: **[Evolving Microstructural Toolkit Data](https://github.com/judmejiabe/evolving-microstructural-toolkit-data)**. 

## 🛠️ Installation

### Option A: Local Install
This method is recommended as it allows you to utilize local GPUs for accelerated training and is best for development. Requires **Python 3.9+** and **R 4.3+**.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/judmejiabe/ttstl_repo.git
    cd ttstl_repo
    ```

2.  **Create Environment:**
    The repository includes two environment files. Choose the one that fits your needs:

    * **Option A.1: Exact Reproduction (Linux/x86_64 only)**
        Use `environment.yml`. This file contains the exact package build strings used in the paper's experiments. It guarantees identical numerical results but is **platform-specific** and may fail on your system.

        ```bash
        conda env create -f environment.yml
        ```

    * **Option A.2: Relaxed Install (Intel macOS/Windows/General)**
        Use `environment_relaxed.yml`. This file maintains the same software versions but removes platform-specific build constraints. *Note: Untested*
        ```bash
        conda env create -f environment_relaxed.yml
        ```
        
    * **Option A.3: 🍎 Apple Silicon (M1/M2/M3) Install**
        This method installs dependencies natively for maximum performance. Since pre-built binaries for some R packages are unavailable on this architecture, this option includes steps to compile them from source.
        1.  Create the base environment:
            ```bash
            conda env create -f environment_native_arm64.yml
            ```
        2.  Activate the environment:
            ```bash
            conda activate ttstl_silicon
            ```
        3.  Install missing R packages from source (requires a working C/Fortran compiler, e.g., Xcode Command Line Tools):
            ```bash
            Rscript -e "remotes::install_version('rugarch', version = '1.5-4', repos='https://cloud.r-project.org'); install.packages('fUnitRoots', repos='https://cloud.r-project.org')"
            ```
        4.  Install the Python package:
            ```bash
            pip install -e .
            ```
    Activate the environment corresponding to your choice in Step 2:
    ```bash
    # If you used Option A.1 (environment.yml)
    conda activate ttstl_dev
    
    # OR if you used Option A.2 (environment_relaxed.yml)
    conda activate ttstl_relaxed
    ```

    Then install the package in editable mode:
    ```bash
    pip install -e .
    ```

### Option B: Docker
We provide a containerized environment with exact Python and R versions to prevent dependency issues.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/judmejiabe/ttstl_repo.git
    cd ttstl_repo
    ```

2.  **Build the image:**
    The Dockerfile is configured to attempt an exact installation first and fall back to a relaxed configuration if necessary.
    ```bash
    docker build -t ttstl .
    ```

## 📋 Usage & Reproducibility

To reproduce the results or run the data generation pipeline, you will need to set up authentication for the required data vendors and cloud storage. This section only outlines general guidelines for reproducibility. For detailed instructions, please refer to the `README.md` file in the dedicated data repository **[Evolving Microstructural Toolkit Data](https://github.com/judmejiabe/evolving-microstructural-toolkit-data)**.

### 1. API Key Generation
The **Data Generation Repository** (`evolving-microstructural-toolkit-data`) requires valid API keys from the following vendors to fetch raw market data.

* **Databento (Market-by-Order Data):**
    1.  Sign up at [databento.com](https://databento.com).
    2.  In the landing page, navigate to **API Keys**.
    3.  Create a new API key. You will need to set it as the `DB_API_KEY` environment variable in the environment used to run the data generation scripts.

* **Massive.com (Previously Polygon.io) (VIX Data):**
    1.  Sign up at [massive.com](https://massive.com).
    2.  Go to your **Dashboard**.
    3.  Create a new API key. You will need to set it as the `POLYGON_API_KEY` environment variable in the environment used to run the data generation scripts.

* **Google Cloud Platform (GCS Storage):**
    1.  Create a new Project in the [Google Cloud Console](https://console.cloud.google.com/).
    2.  Enable the **Google Cloud Storage JSON API** for your project.
    3.  In the Google Cloud Console, go to **IAM & Admin** > **Service Accounts** and create a new service account. Assign the **Storage Admin** role to the service account. 
    4.  Go to the **Manage keys** tab for the service account, click **Add Key** > **Create new key**, and select **JSON**. 
    The JSON key file will automatically download to your computer. Thik JSON key will be necessary to upload and download data to and from GCS. It will be necessary in both environments, the data generation environment and the main environment (i.e. this environment).

### 2. Credentials configuration
The main **TTSTL Repository** uses a `credentials.json` file to authenticate with Google Cloud Storage for fetching processed data. 

**Step 1:** Locate the GCP Service Account JSON key you downloaded in the previous section.

**Step 2:** Rename this file to `credentials.json` and place it in the root directory of the `ttstl_repo`.

**Step 3:** 
    1. In the console, go to **Cloud Storage** > **Buckets** and create a bucket. 
    2. Access the `credentials.json` and add the `bucket_name` field to the JSON object.  This tells the code which bucket to look for.

Your final `credentials.json` should look like this (standard GCP fields + `bucket_name`):

```json
{
  "type": "service_account",
  "project_id": "your-gcp-project-id",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "service-account@your-project.iam.gserviceaccount.com",
  "client_id": "...",
  "auth_uri": "[https://accounts.google.com/o/oauth2/auth](https://accounts.google.com/o/oauth2/auth)",
  "token_uri": "[https://oauth2.googleapis.com/token](https://oauth2.googleapis.com/token)",
  "auth_provider_x509_cert_url": "[https://www.googleapis.com/oauth2/v1/certs](https://www.googleapis.com/oauth2/v1/certs)",
  "client_x509_cert_url": "...",
  "universe_domain": "googleapis.com",
  "bucket_name": "your-gcs-bucket-name"
}
```

**Note:** Ensure `credentials.json` is listed in your `.gitignore` file to prevent accidentally committing your private keys.

### 💻 3. Data Processing Pipeline

To keep the main repository lightweight, all raw data ingestion, heavy processing, and synthetic data generation are handled by our auxiliary repository. This section outlines the workflow for users who wish to **rebuild the datasets from scratch** instead of using the pre-processed versions.

**🔗 Auxiliary Repository:** [Evolving Microstructural Toolkit Data](https://github.com/judmejiabe/evolving-microstructural-toolkit-data)

> **Note:** The scripts mentioned below reside in the *auxiliary repository*, not this one. You must clone that repo and install its dependencies to proceed.

#### Auxilary Repository Environment Configuration

1. Clone the auxiliary repository into a separate directory (e.g., outside `ttstl_repo`).

```bash
git clone https://github.com/judmejiabe/evolving-microstructural-toolkit-data.git
cd evolving-microstructural-toolkit-data
```

2. Set up the environment for the auxiliary repository. We recommend a separate environment to avoid versioning clashes

```bash
conda create -n data_env python=3.10
conda activate data_env
pip install -r requirements.txt
```

3. Set the following `DB_API_KEY` and `POLYGON_API_KEY` environment variables in the environment used to run the data processing scripts:

```bash
conda env config vars set DB_API_KEY=your_databento_key_here POLYGON_API_KEY=your_polygon_api_key_here
conda activate data_env
```

4. Save the `credentials.json` file in the root directory of the `evolving-microstructural-toolkit-data` repository.

#### Synthetic Data Workflow

**Option A: Download Official Data (Recommended)**
We recommend downloading the official dataset from [Zenodo](https://doi.org/10.5281/zenodo.18304963) for exact reproducibility. Download the dataset from [https://doi.org/10.5281/zenodo.18304963](https://doi.org/10.5281/zenodo.18304963). Unzip the downloaded file and place its contents in the `data.nosync` directory at the root of the `evolving-microstructural-toolkit-data` repository.

**Option B: Generate & Upload**
If you wish to generate fresh synthetic data.  Run `Synthetic_Study/synthetic_data_generator.py`. This runs the Markovian simulation of the LOB outlined in the **Synthetic Study** section of the paper to create event-driven and physical-time datasets saved locally.

After executing either of these options, run `Synthetic_Study/upload_synthetic_data.py`. This authenticates using your `credentials.json` and pushes the generated files to the `synthetic_data/` folder in your GCS bucket, making them accessible to the TTSTL orchestrator.

#### Real Market Data Workflow
The processing pipeline transforms raw Databento (MBO) and Polygon (VIX) data into the specific parquet format required by TTSTL.

1.  **Download Raw Data:**
    * Set your environment variables: `DB_API_KEY` (Databento) and `POLYGON_API_KEY` (Polygon).
    * Run `Data_Download/databento_data_download.py` to fetch high-frequency MBO data.
    * Run `Data_Download/polygon_data_download.py` to fetch 1-second VIX aggregates.

2.  **Process & Upload:**
    * Ensure your `credentials.json` (with GCS permissions) is in the root of the data repo.
    * Run `Market_Data_Processing/data_processor.py`.
    * **What it does:** This script parses the MBO messages, constructs the Limit Order Book (LOB), synchronizes it with VIX data, calculates micro-returns, and automatically uploads the final `.parquet` files to your Google Cloud Storage bucket.


## 📂 Repository Structure

* `paper/`: Contains the research paper and supplementary materials.
* `outputs/`: Directory for storing experiment outputs and post-processing results.
    * `analysis_market_data_experiment_results/`: Analysis of market data experiments results.
    * `analysis_synthetic_data_experiment_results/`: Analysis of synthetic data experiments results.
* `src/ttstl/`: Core package source code.
    * `adapter.py`: Adapter module implementation (Volatility modeling & Probabilistic forecasting, see image in the Architecture section).
    * `backbone.py`: Backbone module implementation (Deep learning feature extractor, see image in the Architecture section).
    * `interface.py`: Interface module implementation (Variable selection & R-Python bridging, see image in the Architecture section).
    * `orchestrator.py`: Main API matching backbone, interface, and adapter modules.
    * `architectures/`: Neural network architecture definitions (e.g., CNN-LSTM).
    * `datasets/`: PyTorch dataset implementations for LOB data.
    * `utils/`: Utility functions for data processing and metrics.
    * `r_scripts/`: R scripts wrapped by the modules for statistical methodologies.
* `scripts/`: Experiments and analysis.
    * `analyze_market_data_experiment_results.py`: Post-processing and visualization for market experiments.
    * `analyze_synthetic_data_experiment_results.py`: Post-processing and visualization for synthetic experiments.
    * `market_data_cnn_lstm_pretraining.py`: Script for pretraining the CNN-LSTM backbone on market data.
    * `market_data_experiments.py`: Main script for Intel (INTC) market data experiments.
    * `market_data_ttstl_test.py`: Test script for market data experiments (single date pair).
    * `synthetic_study_analysis.py`: Analysis script for synthetic study results (RelMAE plots, rankings).
    * `synthetic_study_experiment.py`: Main script for synthetic data experiments.
    * `synthetic_study_test.py`: Test script for synthetic data experiments.
    * `utils_experiment_analysis.py`: Unified framework for analyzing experiment results.
* `docs/`: Additional documentation and guides.
* `environment.yml`: Conda environment specification file.
* `Dockerfile`: Docker container build configuration.
* `CITATION.cff`: Citation metadata.
* `credentials.example.json`: Template for GCP credentials.
* `LICENSE`: License text.
* `pyproject.toml`: Build system requirements and dependencies.
* `README.md`: The file you are reading.


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