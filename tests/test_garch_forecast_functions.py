#!/usr/bin/env python3
"""
GARCH Analysis Replication using Individual R Function Calls
================================================================

This script replicates the R GARCH analysis using individual R function calls 
through rpy2 (similar to the adapter module) instead of executing the full R code block.
This approach allows for verification of the adapter forecast method implementation.

Data preprocessing (aggregation, filtering, and standardization) uses pandas,
while GARCH modeling uses R's rugarch package through rpy2 for individual function calls.

The script maintains the same covariates as specified in the original R script:
- OFI0 (Order Flow Imbalance)
- traded_volume 
- n_trades
- mean_trades

Author: GitHub Copilot
Date: September 19, 2025
"""
import logging
import os
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
matplotlib.use('Agg')  # Use non-interactive backend

# Activate automatic pandas-R conversion
pandas2ri.activate()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_r_packages():
    """Load required R packages"""
    try:
        # Import R packages
        rugarch = importr('rugarch')

        logger.info("Successfully loaded R packages: rugarch")
        return rugarch
    except Exception as e:
        logger.error(f"Failed to load R packages: {e}")
        raise RuntimeError(f"Failed to load R packages: {e}")

def load_and_process_data(data_file_path):
    """Load and process the data file"""
    logger.info(f"Loading data from: {data_file_path}")
    
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")
    
    # Load data
    data = pd.read_csv(data_file_path)
    logger.info(f"Loaded data shape: {data.shape}")
    
    # Calculate micro_ret as log return of micro price
    data['micro_ret'] = np.log(data['micro']).diff() * 1_000
    
    # Calculate mean_trades as traded_volume / n_trades (avoid division by zero)
    data['mean_trades'] = np.where(data['n_trades'] > 0, 
                                  data['traded_volume'] / data['n_trades'], 
                                  0)
    
    # Convert timestamp to numeric (nanoseconds since epoch)
    data['timestamp_ns'] = pd.to_datetime(data['timestamp']).astype(np.int64)
    
    # Log time window
    min_time = data['timestamp_ns'].min()
    max_time = data['timestamp_ns'].max()
    logger.info(f"Time window: {min_time} to {max_time} ns")
    
    return data

def create_2sec_bins(data):
    """Create 2-second time bins"""
    # Create 2-second bins using R-style approach
    data['time_2sec_bin'] = (data['timestamp_ns'] // 2_000_000_000) * 2_000_000_000
    
    unique_bins = data['time_2sec_bin'].nunique()
    logger.info(f"Created 2-second bins, unique bins: {unique_bins}")
    
    return data

def aggregate_to_bins_pandas(data):
    """Aggregate data to 2-second bins using pandas"""
    logger.info("Aggregating to 2-second bins using pandas...")
    
    # Create a copy of the data
    data_copy = data.copy()
    
    # Filter for trading hours using integer nanosecond timestamps
    # 9:30 AM to 4:00 PM EST on 2024-06-03
    # Convert to UTC: 9:30 AM EST = 1:30 PM UTC, 4:00 PM EST = 8:00 PM UTC
    start_time_ns = 35100 * 1_000_000_000  # 9:45 AM EST in nanoseconds
    end_time_ns = 37500 * 1_000_000_000  # 10:25 AM EST in nanoseconds

    # Filter for trading hours using the original timestamp_ns column
    data_filtered = data_copy[
        (data_copy['timestamp_ns'] >= start_time_ns) &
        (data_copy['timestamp_ns'] < end_time_ns)
    ].copy()
    
    logger.info(f"After time filtering: {len(data_filtered)} observations")
    
    if len(data_filtered) == 0:
        logger.warning("No data remaining after time filtering!")
        return pd.DataFrame()
    
    # Aggregate by 2-second bins using pandas groupby
    aggregated_data = data_filtered.groupby('time_2sec_bin').agg({
        'timestamp': 'first',
        'timestamp_ns': 'min',
        'time_2sec_bin': 'first',
        'micro_ret': 'sum',
        'OFI0': 'sum',
        'traded_volume': 'sum',
        'n_trades': 'sum'
    }).reset_index(drop=True)
    
    # Compute mean_trades after aggregation, handling zero n_trades
    aggregated_data['mean_trades'] = np.where(
        aggregated_data['n_trades'] > 0,
        aggregated_data['traded_volume'] / aggregated_data['n_trades'],
        0
    )
    
    # Sort by time_2sec_bin
    aggregated_data = aggregated_data.sort_values('time_2sec_bin').reset_index(drop=True)
    
    logger.info(f"After aggregation and filtering: {len(aggregated_data)} bins")
    
    if len(aggregated_data) > 0:
        logger.info(f"Time range: {aggregated_data['time_2sec_bin'].min()} to {aggregated_data['time_2sec_bin'].max()}")
    
    return aggregated_data

def standardize_covariates_pandas(data, covariate_names):
    """Standardize covariates using pandas (z-score normalization)"""
    logger.info(f"Standardizing covariates: {covariate_names}")
    
    # Create a copy of the data to avoid modifying the original
    standardized_data = data.copy()
    
    # Standardize each covariate using z-score normalization
    for var in covariate_names:
        if var in standardized_data.columns:
            # Calculate mean and standard deviation
            mean_val = standardized_data[var].mean()
            std_val = standardized_data[var].std()
            
            # Apply z-score normalization: (x - mean) / std
            if std_val > 0:  # Avoid division by zero
                standardized_data[var] = (standardized_data[var] - mean_val) / std_val
                logger.info(f"Standardized {var}: mean={mean_val:.4f}, std={std_val:.4f}")
            else:
                logger.warning(f"Variable {var} has zero standard deviation, keeping original values")
        else:
            logger.warning(f"Variable {var} not found in data columns")
    
    logger.info("Covariates standardized using pandas")
    return standardized_data

def create_visualizations(data):
    """Create visualizations of the data"""
    logger.info("Creating visualizations...")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Micro Returns and Covariates Analysis', fontsize=16)
    
    # Plot micro returns
    axes[0, 0].plot(data.index, data['micro_ret'])
    axes[0, 0].set_title('Micro Returns')
    axes[0, 0].set_ylabel('Micro Return')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot OFI0
    axes[0, 1].plot(data.index, data['OFI0'], color='orange')
    axes[0, 1].set_title('Order Flow Imbalance (OFI0)')
    axes[0, 1].set_ylabel('OFI0 (standardized)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot traded volume
    axes[0, 2].plot(data.index, data['traded_volume'], color='green')
    axes[0, 2].set_title('Traded Volume')
    axes[0, 2].set_ylabel('Volume (standardized)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot n_trades
    axes[1, 0].plot(data.index, data['n_trades'], color='red')
    axes[1, 0].set_title('Number of Trades')
    axes[1, 0].set_ylabel('N Trades (standardized)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot mean_trades
    axes[1, 1].plot(data.index, data['mean_trades'], color='purple')
    axes[1, 1].set_title('Mean Trades')
    axes[1, 1].set_ylabel('Mean Trades (standardized)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Correlation heatmap
    corr_vars = ['micro_ret', 'OFI0', 'traded_volume', 'n_trades', 'mean_trades']
    corr_matrix = data[corr_vars].corr()
    
    im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 2].set_title('Correlation Matrix')
    axes[1, 2].set_xticks(range(len(corr_vars)))
    axes[1, 2].set_yticks(range(len(corr_vars)))
    axes[1, 2].set_xticklabels(corr_vars, rotation=45, ha='right')
    axes[1, 2].set_yticklabels(corr_vars)
    
    # Add correlation values to heatmap
    for i in range(len(corr_vars)):
        for j in range(len(corr_vars)):
            text = axes[1, 2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=axes[1, 2])
    plt.tight_layout()
    
    # Save in tests directory
    tests_dir = os.path.dirname(__file__)
    plot_path = os.path.join(tests_dir, 'micro_returns_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved visualization: {plot_path}")
    
    # Print correlation matrix
    logger.info("Correlation Matrix:")
    logger.info(f"\n{corr_matrix}")
    
    return corr_matrix

def run_garch_analysis_individual_calls(data, rugarch):
    """Run eGARCH analysis using run_garch R function (like adapter)"""
    logger.info("Running eGARCH analysis using run_garch R function...")
    
    # Prepare data
    y = data['micro_ret'].values
    external_regressors = data[['OFI0', 'traded_volume', 'n_trades', 'mean_trades']].values
    logger.info(f"External regressors head: {external_regressors[0:20]}")
    
    total_obs = len(y)
    n_fit = 600  # In-sample training observations
    n_out = total_obs - n_fit  # Out-of-sample observations
    
    logger.info(f"Total observations: {total_obs}")
    logger.info(f"Training observations: {n_fit}")
    logger.info(f"Out-of-sample observations: {n_out}")
    
    # Set working directory to tests folder for R plots
    tests_dir = os.path.dirname(__file__)
    ro.r(f'setwd("{tests_dir}")')
    logger.info(f"Set R working directory to: {tests_dir}")
    
    # Load the R script with run_garch function (same as adapter)
    r_script_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'ttstl', 'r_scripts', 'run_garch.R')
    ro.r(f'source("{r_script_path}")')
    run_garch_model = ro.globalenv['run_garch']
    
    logger.info("Loaded run_garch R function")
    
    # Convert external regressors to pandas DataFrame for R conversion
    #X = pd.DataFrame(external_regressors, columns=['OFI0', 'traded_volume', 'n_trades', 'mean_trades'])
    #logger.info(f"X head:\n{X[0:20]}")
    
    # Call run_garch function (same parameters as adapter)
    logger.info("Fitting eGARCH model using run_garch...")
    
    # Create plot label with timestamp to avoid overwrites
    plot_label = f"unit_test"
    
    r_result = run_garch_model(
        y=y,
        X=external_regressors[0:n_fit, :],  # External regressors for mean model
        Z=ro.NULL,  # No external regressors for variance model
        p=1,  # ARMA(1,1)
        q=1,
        m=1,  # eGARCH(1,1)
        s=1,
        archm=False,
        dist='nig',  # Use normal inverse Gaussian distribution
        model='eGARCH',
        start_pars=ro.ListVector({}),  # No starting parameters
        out_sample=n_out,
        display=True,
        plot_label=plot_label
    )
    
    logger.info("GARCH model fitted successfully")
    
    # Extract results from R (same as adapter)
    garch_results = {}
    garch_results['coefficients'] = np.array(r_result.rx2('coef'))
    garch_results['se_coefficients'] = np.array(r_result.rx2('se_coef'))
    garch_results['p_values'] = np.array(r_result.rx2('pvals'))
    garch_results['coefficient_names'] = list(r_result.rx2('coef_names'))
    garch_results['gof_pval'] = np.array(r_result.rx2('gof'))
    garch_results['lb_pval'] = np.array(r_result.rx2('lb'))
    garch_results['lb_sq_pval'] = np.array(r_result.rx2('lb_sq'))
    garch_results['standardized_residuals'] = np.array(r_result.rx2('stdresid'))
    garch_results['saddle_point_indicator'] = int(r_result.rx2('saddle')[0])
    
    # Get the fitted model for forecasting
    garch_fit = r_result.rx2('fit')
    
    logger.info("GARCH Model Results:")
    logger.info(f"Coefficients: {garch_results['coefficient_names']}")
    for i, (name, coef, se, pval) in enumerate(zip(
        garch_results['coefficient_names'], 
        garch_results['coefficients'], 
        garch_results['se_coefficients'], 
        garch_results['p_values']
    )):
        logger.info(f"  {name}: {coef:.6f} (SE: {se:.6f}, p-value: {pval:.4f})")

    logger.info(f"Goodness-of-fit p-values:\n {garch_results['gof_pval']}")
    
    # Step 4: Generate forecasts using ugarchforecast
    logger.info("Generating rolling forecasts...")
    
    # Prepare external forecasts for out-of-sample period
    external_forecasts = ro.ListVector({
        'mregfor': ro.r['matrix'](external_regressors[n_fit:, :],
                                 nrow=total_obs - n_fit,
                                 ncol=external_regressors.shape[1])
    })
    
    # Generate rolling forecasts
    garch_forecast = rugarch.ugarchforecast(
        fitORspec=garch_fit,
        n_ahead=1,
        n_roll=n_out - 1,
        external_forecasts=external_forecasts
    )
    
    logger.info("Rolling forecasts generated successfully")
    
    # Step 5: Extract forecast results
    
    # Get fitted values and forecasts
    forecast_mean = rugarch.fitted(garch_forecast)
    forecast_vol = rugarch.sigma(garch_forecast)
    
    # Convert to numpy arrays
    forecast_mean_array = np.array(forecast_mean).flatten()
    forecast_vol_array = np.array(forecast_vol).flatten()
    
    # Get actual out-of-sample values
    actual_values = y[-n_out:]
    
    # Calculate R²
    ss_res = np.sum((actual_values - forecast_mean_array) ** 2)
    ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    logger.info(f"Out-of-sample R²: {r2:.4f}")
    
    # Step 6: Sign prediction analysis
    actual_signs = actual_values > 0
    forecast_signs = forecast_mean_array > 0
    
    # Create confusion matrix
    confusion_matrix = pd.crosstab(
        pd.Series(forecast_signs, name='Predicted'),
        pd.Series(actual_signs, name='Actual'),
        margins=True
    )
    
    logger.info("Sign Prediction Confusion Matrix:")
    print(confusion_matrix)
    
    return {
        'garch_fit': garch_fit,
        'garch_forecast': garch_forecast,
        'garch_results': garch_results,
        'forecast_mean': forecast_mean_array,
        'forecast_vol': forecast_vol_array,
        'actual_values': actual_values,
        'r2': r2,
        'confusion_matrix': confusion_matrix,
        'n_fit': n_fit,
        'n_out': n_out
    }

def run_garch_analysis_adapter_approach(data, rugarch):
    """Run eGARCH analysis using adapter's approach: fit with fixed parameters + ugarchforecast"""
    logger.info("Running eGARCH analysis using adapter's approach (fixed parameters + ugarchforecast)...")
    
    # Prepare data
    y = data['micro_ret'].values
    external_regressors = data[['OFI0', 'traded_volume', 'n_trades', 'mean_trades']].values
    
    total_obs = len(y)
    n_fit = 600  # In-sample training observations
    n_out = total_obs - n_fit  # Out-of-sample observations
    
    logger.info(f"Adapter approach - Total observations: {total_obs}")
    logger.info(f"Adapter approach - Training observations: {n_fit}")
    logger.info(f"Adapter approach - Out-of-sample observations: {n_out}")
    
    # Step 1: Fit the initial model to get parameters (similar to adapter.update_adapter)
    logger.info("Step 1: Fitting initial GARCH model to obtain parameters...")
    
    # Load R script functions
    r_script_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'ttstl', 'r_scripts', 'run_garch.R')
    ro.r(f'source("{r_script_path}")')
    run_garch_model = ro.globalenv['run_garch']
    set_fixed_parameters = ro.globalenv['set_fixed_parameters']
    
    # Initial fit on training data only (no out_sample for parameter estimation)
    r_result = run_garch_model(
        y=y[:n_fit],  # Only training data for parameter estimation
        X=external_regressors[:n_fit, :],
        Z=ro.NULL,
        p=1, q=1, m=1, s=1,
        archm=False,
        dist='nig',
        model='eGARCH',
        start_pars=ro.ListVector({}),
        out_sample=0,  # No out-of-sample in initial fit
        display=False
    )
    
    # Extract the fitted parameters and specification
    fitted_spec = r_result.rx2('spec')
    fitted_coefficients = r_result.rx2('coef')
    coefficient_names = list(r_result.rx2('coef_names'))
    
    logger.info("Step 1 complete: Extracted fitted parameters")
    logger.info(f"Fitted coefficients: {coefficient_names}")
    
    # Step 2: Create new spec with fixed parameters (adapter's approach)
    logger.info("Step 2: Creating specification with fixed parameters...")
    
    # Convert coefficients to R named list for setting fixed parameters
    coef_dict = {name: float(fitted_coefficients[i]) for i, name in enumerate(coefficient_names)}
    fixed_params_r = ro.ListVector(coef_dict)
    
    # Set fixed parameters in the specification
    spec_with_fixed_pars = set_fixed_parameters(fitted_spec, fixed_params_r)
    
    logger.info("Step 2 complete: Created specification with fixed parameters")
    
    # Step 3: Perform rolling forecast using ugarchforecast directly (adapter's approach)
    logger.info("Step 3: Performing rolling forecast with ugarchforecast...")
    
    # Prepare external forecasts for the entire dataset (including out-of-sample)
    external_forecasts = ro.ListVector({
        'mregfor': ro.r['matrix'](
            external_regressors[n_fit:, :],
            nrow=n_out,
            ncol=external_regressors.shape[1]
        )
    })
    
    # Use ugarchforecast with the fixed parameter specification
    garch_forecast = rugarch.ugarchforecast(
        fitORspec=spec_with_fixed_pars,
        data=y,
        n_ahead=1,
        n_roll=n_out - 1,
        out_sample=n_out,
        external_forecasts=external_forecasts
    )
    
    logger.info("Step 3 complete: Rolling forecasts generated")
    
    # Step 4: Extract forecast results
    forecast_mean = rugarch.fitted(garch_forecast)
    forecast_vol = rugarch.sigma(garch_forecast)
    
    # Convert to numpy arrays
    forecast_mean_array = np.array(forecast_mean).flatten()
    forecast_vol_array = np.array(forecast_vol).flatten()
    
    # Get actual out-of-sample values
    actual_values = y[-n_out:]
    
    # Calculate R²
    ss_res = np.sum((actual_values - forecast_mean_array) ** 2)
    ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    logger.info(f"Adapter approach - Out-of-sample R²: {r2:.4f}")
    
    # Extract model information for comparison
    garch_results = {}
    garch_results['coefficients'] = np.array(fitted_coefficients)
    garch_results['coefficient_names'] = coefficient_names
    garch_results['fixed_parameters'] = coef_dict
    
    logger.info("Adapter Approach - Model Results:")
    logger.info(f"Fixed Parameters: {coefficient_names}")
    for i, (name, coef) in enumerate(zip(coefficient_names, garch_results['coefficients'])):
        logger.info(f"  {name}: {coef:.6f} (FIXED)")
    
    # Sign prediction analysis
    actual_signs = actual_values > 0
    forecast_signs = forecast_mean_array > 0
    
    confusion_matrix = pd.crosstab(
        pd.Series(forecast_signs, name='Predicted'),
        pd.Series(actual_signs, name='Actual'),
        margins=True
    )
    
    logger.info("Adapter Approach - Sign Prediction Confusion Matrix:")
    print(confusion_matrix)
    
    return {
        'garch_forecast': garch_forecast,
        'garch_results': garch_results,
        'forecast_mean': forecast_mean_array,
        'forecast_vol': forecast_vol_array,
        'actual_values': actual_values,
        'r2': r2,
        'confusion_matrix': confusion_matrix,
        'n_fit': n_fit,
        'n_out': n_out,
        'method': 'adapter_approach'
    }

def compare_approaches(data, rugarch):
    """Compare the current approach vs adapter's approach"""
    logger.info("=" * 80)
    logger.info("COMPARING APPROACHES: Current vs Adapter")
    logger.info("=" * 80)
    
    # Run current approach
    logger.info("Running CURRENT approach (run_garch with out_sample)...")
    current_results = run_garch_analysis_individual_calls(data, rugarch)
    
    # Run adapter approach  
    logger.info("Running ADAPTER approach (fixed parameters + ugarchforecast)...")
    adapter_results = run_garch_analysis_adapter_approach(data, rugarch)
    
    # Compare results
    logger.info("=" * 80)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 80)
    
    logger.info(f"Current approach R²: {current_results['r2']:.6f}")
    logger.info(f"Adapter approach R²: {adapter_results['r2']:.6f}")
    logger.info(f"R² difference: {abs(current_results['r2'] - adapter_results['r2']):.6f}")
    
    # Compare forecast means
    current_mean = current_results['forecast_mean']
    adapter_mean = adapter_results['forecast_mean']
    
    mean_diff = np.mean(np.abs(current_mean - adapter_mean))
    max_diff = np.max(np.abs(current_mean - adapter_mean))
    correlation = np.corrcoef(current_mean, adapter_mean)[0, 1]
    
    logger.info(f"Forecast mean differences:")
    logger.info(f"  Mean absolute difference: {mean_diff:.6f}")
    logger.info(f"  Maximum absolute difference: {max_diff:.6f}")
    logger.info(f"  Correlation: {correlation:.6f}")
    
    # Compare forecast volatilities
    current_vol = current_results['forecast_vol']
    adapter_vol = adapter_results['forecast_vol']
    
    vol_mean_diff = np.mean(np.abs(current_vol - adapter_vol))
    vol_max_diff = np.max(np.abs(current_vol - adapter_vol))
    vol_correlation = np.corrcoef(current_vol, adapter_vol)[0, 1]
    
    logger.info(f"Forecast volatility differences:")
    logger.info(f"  Mean absolute difference: {vol_mean_diff:.6f}")
    logger.info(f"  Maximum absolute difference: {vol_max_diff:.6f}")
    logger.info(f"  Correlation: {vol_correlation:.6f}")
    
    # Compare confusion matrices
    logger.info("Confusion Matrix Comparison:")
    logger.info("Current approach:")
    print(current_results['confusion_matrix'])
    logger.info("Adapter approach:")
    print(adapter_results['confusion_matrix'])
    
    return {
        'current_results': current_results,
        'adapter_results': adapter_results,
        'comparison_metrics': {
            'r2_difference': abs(current_results['r2'] - adapter_results['r2']),
            'mean_forecast_correlation': correlation,
            'vol_forecast_correlation': vol_correlation,
            'mean_absolute_diff': mean_diff,
            'vol_absolute_diff': vol_mean_diff
        }
    }

def create_forecast_visualizations(results):
    """Create forecast result visualizations"""
    logger.info("Creating forecast visualizations...")
    
    forecast_mean = results['forecast_mean']
    forecast_vol = results['forecast_vol']
    actual_values = results['actual_values']
    n_out = results['n_out']
    
    # Create time index for out-of-sample period
    time_index = np.arange(n_out)
    
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('GARCH Forecast Results', fontsize=16)
    
    # Plot 1: Actual vs Predicted Returns
    axes[0].plot(time_index, actual_values, label='Actual', alpha=0.7, linewidth=1)
    axes[0].plot(time_index, forecast_mean, label='Forecast', alpha=0.8, linewidth=1.5)
    axes[0].set_title('Actual vs Forecasted Returns')
    axes[0].set_ylabel('Returns')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Forecast Volatility
    axes[1].plot(time_index, forecast_vol, color='red', alpha=0.8)
    axes[1].set_title('Forecasted Volatility')
    axes[1].set_ylabel('Volatility')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot of Actual vs Predicted
    axes[2].scatter(forecast_mean, actual_values, alpha=0.5, s=1)
    axes[2].plot([forecast_mean.min(), forecast_mean.max()], 
                [forecast_mean.min(), forecast_mean.max()], 
                'r--', alpha=0.8, label='Perfect Prediction')
    axes[2].set_xlabel('Predicted Returns')
    axes[2].set_ylabel('Actual Returns')
    axes[2].set_title(f'Actual vs Predicted (R² = {results["r2"]:.4f})')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save in tests directory
    tests_dir = os.path.dirname(__file__)
    plot_path = os.path.join(tests_dir, 'garch_forecast_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved forecast visualization: {plot_path}")


def main():
    """Main function to run the GARCH analysis"""
    logger.info("=" * 80)
    logger.info("GARCH Analysis Replication using Individual R Function Calls")
    logger.info("=" * 80)
    
    try:
        # Load R packages
        rugarch = load_r_packages()
        
        # Define data file path
        data_file_path = Path(__file__).parent / "test_data" / "06032024_INTC_physical_time_driven_recorded_data.csv"
        
        # Load and process data
        data = load_and_process_data(data_file_path)
        
        # Create 2-second bins
        data = create_2sec_bins(data)
        
        # Aggregate to bins using pandas
        aggregated_data = aggregate_to_bins_pandas(data)

        logger.info(f"Aggregated data shape: {aggregated_data.shape}")
        logger.info(f"Aggregated data info: {aggregated_data.info()}")
        
        # Standardize covariates using pandas
        covariate_names = ['OFI0', 'traded_volume', 'n_trades', 'mean_trades']
        standardized_data = standardize_covariates_pandas(aggregated_data, covariate_names)
        logger.info(f"Standardized data head: {standardized_data[['timestamp_ns', 'OFI0', 'micro_ret']][0:20]}")
        
        # Create visualizations
        correlation_matrix = create_visualizations(standardized_data)
        
        # Run GARCH analysis using individual R function calls
        results = run_garch_analysis_individual_calls(standardized_data, rugarch)
        
        # Compare approaches: current vs adapter
        logger.info("\n" + "=" * 80)
        logger.info("COMPARING CURRENT APPROACH VS ADAPTER APPROACH")
        logger.info("=" * 80)
        comparison_results = compare_approaches(standardized_data, rugarch)
        
        # Create forecast visualizations
        create_forecast_visualizations(results)
        
        
        logger.info("=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Current approach R²: {results['r2']:.4f}")
        logger.info(f"Adapter approach R²: {comparison_results['adapter_results']['r2']:.4f}")
        logger.info(f"R² difference: {comparison_results['comparison_metrics']['r2_difference']:.6f}")
        logger.info("Generated files:")
        logger.info("  - micro_returns_analysis.png")
        logger.info("  - garch_forecast_results.png")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()