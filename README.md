![CI/CD Pipeline](https://github.com/judmejiabe/ttstl_repo/actions/workflows/ci.yml/badge.svg)

# TTSTL: Two Time-Scale Transfer Learning
 (TTSTL) Framework

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
* **MBO Data:** Raw market-by-order data must be licensed directly from **[Databento](https://databento.com)**. As of January 2026, **Databento** offers $125 USD in free credits towards subscription plans, which is sufficient to acquire the data required for this paper's experiments.
* **VIX Data:** Volatility index data must be sourced from **[Massive.com](https://massive.com)** (previously known as **[Polygon.io](https://polygon.io)**). This service requires a subscription plan.
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
        Use `environment.yml`. This file contains the exact package build strings used in the paper's experiments. It guarantees identical numerical results to those in the paper but is **platform-specific** and may fail on different operating systems.

        ```bash
        conda env create -f environment.yml
        ```

    * **Option A.2: Relaxed Install (Intel macOS/Windows/General)**
        Use `environment_relaxed.yml`. This file maintains the same software versions but removes platform-specific build constraints. *Note: This environment allows for broader compatibility but has not been validated against the paper's exact numerical output.*
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

This section outlines the steps required to reproduce the paper's results.

### 🔑 1. API Key Generation
The **Data Generation Repository** (`evolving-microstructural-toolkit-data`) requires valid API keys from the following vendors to fetch raw market data.

* **Databento (Market-by-Order Data):**
    1.  Sign up at [databento.com](https://databento.com).
    2.  Navigate to **API Keys** on the landing page.
    3.  Create a new API key. You will need to set it as the `DB_API_KEY` environment variable in the environment used to run the data generation scripts.
    4.  You must provide a payment method to access the MBO data. Databento's MBO data is a pay-as-you-go service. As of January 21st, 2026, **Databento** offers $125 USD in free credits towards subscription plans, which is sufficient to acquire the data required for this paper's experiments.

* **Massive.com (Previously Polygon.io) (VIX Data):**
    1.  Sign up at [massive.com](https://massive.com).
    2.  Go to your **Dashboard**.
    3.  Create a new API key. You will need to set it as the `POLYGON_API_KEY` environment variable in the environment used to run the data generation scripts.
    4.  You will need a subscription to the **indices starter** plan or superior to access the VIX data.

* **Google Cloud Platform (GCS Storage):**
    1.  Create a new Project in the [Google Cloud Console](https://console.cloud.google.com/).
    2.  Enable the **Google Cloud Storage JSON API** for your project.
    3.  In the Google Cloud Console, go to **IAM & Admin** > **Service Accounts** and create a new service account. Assign the **Storage Admin** role to the service account. 
    4.  Go to the **Manage keys** tab for the service account, click **Add Key** > **Create new key**, and select **JSON**. 
    The JSON key file will automatically download to your computer. This JSON key is required to upload and download data to and from GCS. It is needed in both the data generation environment and the main environment (i.e., this environment).

### 🔐 2. Credentials Configuration

The main **TTSTL Repository** uses a `credentials.json` file to authenticate with Google Cloud Storage for fetching processed data. 

**Step 1:** Locate the GCP Service Account JSON key you downloaded in the previous section.

**Step 2:** Rename this file to `credentials.json` and place it in the root directory of the `ttstl_repo`.

**Step 3:** 
    1. In the console, go to **Cloud Storage** > **Buckets** and create a bucket. 
    2. Open `credentials.json` and append the `bucket_name` field to the JSON object. This configuration instructs the application which GCS bucket to target.

Your final `credentials.json` should look like this (standard GCP fields + `bucket_name`):

```json
{
  "type": "service_account",
  "project_id": "your-gcp-project-id",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "service-account@your-project.iam.gserviceaccount.com",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "...",
  "universe_domain": "googleapis.com",
  "bucket_name": "your-gcs-bucket-name"
}
```

**Note:** Ensure `credentials.json` is listed in your `.gitignore` file to prevent accidentally committing your private keys.

### 💻 3. Data Processing Pipeline

To maintain a lightweight core repository, we offload raw data ingestion, heavy processing, and synthetic data generation to a dedicated auxiliary repository. This section outlines the workflow for users wishing to **rebuild the datasets from scratch** rather than using the pre-processed versions.

**🔗 Auxiliary Repository:** [Evolving Microstructural Toolkit Data](https://github.com/judmejiabe/evolving-microstructural-toolkit-data)

> **Note:** The scripts mentioned below reside in the *auxiliary repository*, not this one. You must clone that repo and install its dependencies to proceed.

#### Auxiliary Repository Environment Configuration

1. Clone the auxiliary repository into a separate directory (e.g., outside `ttstl_repo`).

```bash
git clone https://github.com/judmejiabe/evolving-microstructural-toolkit-data.git
cd evolving-microstructural-toolkit-data
```

2. Set up the environment for the auxiliary repository. We recommend a separate environment to avoid versioning conflicts.

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

**Option A: Download Original Synthetic Data (Recommended)**
We recommend downloading the original synthetic dataset from [Zenodo](https://doi.org/10.5281/zenodo.18304963) for exact reproducibility. Unzip the downloaded file and place its contents in the `data.nosync/Recorded_Synthetic_Data` directory at the root of the `evolving-microstructural-toolkit-data` repository.

**Option B: Generate Fresh Synthetic Data**

> **Note:** Generating fresh data will produce different datasets than those provided on Zenodo. This process is computationally intensive and can take up to 24 hours; we recommend running it in the background. If you choose this option, ensure you update the file names in `scripts/synthetic_study_experiment.py` (e.g., `08042025_long_event_driven_recorded_lob_{id}.parquet`) to match the files in your GCS bucket.

To generate fresh synthetic data, run `Synthetic_Study/synthetic_data_generator.py`. This executes the Markovian simulation of the LOB outlined in the **Synthetic Study** section of the paper, creating locally saved event-driven and physical-time datasets. This will generate different datasets than the ones provided in the Zenodo repository.

After executing either of these options, run `Synthetic_Study/upload_synthetic_data.py`. This authenticates using your `credentials.json` and pushes the generated files to the `synthetic_data/` folder in your GCS bucket, making them accessible to the TTSTL orchestrator.

#### Real Market Data Workflow
The processing pipeline transforms raw Databento (MBO) and Massive (VIX) data into the specific parquet format required by TTSTL.

1.  **Download Raw Data:**
    * Verify that your Databento and Massive accounts are configured properly and that the `DB_API_KEY` and `POLYGON_API_KEY` environment variables are set.
    * Run `Data_Download/polygon_data_download.py` to fetch 1-second VIX aggregates. The data will download to the `data.nosync/Polygon/Downloads` directory.
    * Run `Data_Download/databento_data_download.py` to fetch high-frequency MBO data. The data will download to the `data.nosync/Databento/Downloads` directory.
    

2.  **Process & Upload:**
    * Ensure your `credentials.json` (with GCS permissions) is in the root of the data repo.
    * Run `Market_Data_Processing/data_processor.py`. This script parses the MBO messages, constructs the Limit Order Book (LOB), synchronizes it with VIX data, calculates micro-returns, and automatically uploads the final `.parquet` files to your Google Cloud Storage bucket. This process is computationally expensive and can take multiple hours to complete. We recommend running this process in the background.

### 🧪 4. Verification

We provide tools to verify both the installation integrity and the downloaded data.

#### Installation Verification
We provide a test suite to verify the installation and codebase integrity. The tests are divided into two tiers:

*   **Lightweight Tests**: Fast sanity checks to ensure the environment path and basic imports are working.
*   **Comprehensive Tests**: Full-pipeline experiments that replicate the paper's results to ensure end-to-end correctness. These can take 30+ minutes.

**Run lightweight tests (Recommended):**
```bash
pytest -m lightweight
```

**Run comprehensive suite:**
```bash
pytest -m comprehensive
```

#### Data Verification (Optional)
Use the `scripts/validate_data.py` tool to verify your downloaded files against the official `data_manifest.json`. This ensures that your local data matches the expected state.

**Prerequisites:**
1.  **Download Data**: Ensure you have downloaded the datasets as described in the [Data Processing Pipeline](#-3-data-processing-pipeline) section.
2.  **Directory Structure**: The script expects the following directory structure inside your `data` folder:
    *   `data/synthetic_data/`: Contains the synthetic `.parquet` files.
    *   `data/intc_data/`: Contains the INTC market data `.parquet` files.

**Configuration:**
By default, the script looks for the `data` directory at the project root. If your data is located elsewhere, you **must** set the `TTSTL_DATA_DIR` environment variable:

```bash
export TTSTL_DATA_DIR="/path/to/your/custom/data_dir"
```

**Usage:**
```bash
# Validate all datasets
python scripts/validate_data.py --dataset all

# If you prefer to validate a specific dataset
python scripts/validate_data.py --dataset synthetic
python scripts/validate_data.py --dataset intc
```

**⚠️ Warning:**
The verification process is very sensitive to the software environment. Even minor differences, such as a mismatch in the `pyarrow` library version (e.g., between what was used to serialize the data and your local version), can cause the SHA-256 hash to differ, resulting in a test failure.
*   **Success**: "Verification Successful: All files matched the manifest."
*   **Failure**: "Corruption Detected." This occurs if files are missing, corrupted, or if your environment differs from the one used to generate the manifest.

> **Note:** You do not need to download the data to run the experiments. The experiment scripts automatically download the data from the GCS bucket configured in your `credentials.json`.

> **Note:** If you have generated fresh synthetic data, verification failure is expected as your local files will naturally differ from the official manifest.



### 🧪🧪 5. Reproduction Steps

#### Synthetic Study

1.  **Run Experiments**: Execute the following command to run the 28 configurations described in the paper on the synthetic dataset:

    > **Note:** If fresh synthetic data was generated, ensure the file names in `scripts/synthetic_study_experiment.py` match the `.parquet` files in your GCS bucket (e.g., `08042025_long_event_driven_recorded_lob_{id}.parquet`).

    ```bash
    python scripts/synthetic_study_experiment.py
    ```
    *   **Output**: Experiment results and model checkpoints are saved in `outputs/runs/`.
    *   **Logs**: Comprehensive logs are saved to `logs/synthetic_study_<timestamp>.log`.

2.  **Analyze Results**: Execute the analysis script to generate the rankings (Table 2) and RelMAE plots:
    ```bash
    python scripts/synthetic_study_analysis.py
    ```
    *   **Output**: Plots and rankings are saved in `outputs/analysis_synthetic_data_experiment_results/`.

#### Market Data Study

1.  **Pretrain Backbone**: Train the CNN-LSTM backbone on May 2024 data:
    ```bash
    python scripts/market_data_cnn_lstm_pretraining.py
    ```
    *   **Output**: Pretrained models are saved in `outputs/models/`.

2.  **Run Experiments**: Run the recursive forecasting experiments on the INTC dataset using the pretrained backbone:
    ```bash
    python scripts/market_data_experiments.py
    ```
    *   **Output**: Experiment results are saved in `outputs/runs/`.
    *   **Logs**: Comprehensive logs are saved to `logs/market_data_experiments_<type>_<timestamp>.log`.

3.  **Analyze Results**: Generate the performance boxplots and statistics:
    ```bash
    python scripts/analyze_market_data_experiment_results.py
    ```
    *   **Output**: Boxplots and statistics are saved in `outputs/analysis_market_data_experiment_results/`.

## 🔗 Citation
If you use this code in your research, please cite the following working paper:

```bibtex
@article{mejiabecerra2026twotimescale,
  title = {Two-Time-Scale Transfer Learning for Market-by-Order Micro-Return Forecasting},
  author = {Mejia Becerra, Juan Diego and Peters, Gareth W. and Hu, Ruimeng},
  year = {2026},
  journal = {SSRN Electronic Journal},
  doi = {10.2139/ssrn.6424798},
  url = {https://ssrn.com/abstract=6424798}
}
```