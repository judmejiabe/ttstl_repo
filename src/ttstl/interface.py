import math
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy import stats
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_white

from ttstl.backbone import Backbone

# Activate automatic conversion between pandas/numpy and R
pandas2ri.activate()
numpy2ri.activate()

# Import R libraries
try:
    glmnet = importr('glmnet')
    rugarch = importr('rugarch')
    stats_r = importr('stats')
    base = importr('base')
except Exception as e:
    warnings.warn(f"Could not import R packages: {e}. Make sure glmnet and rugarch are installed in R.")
    glmnet = None
    rugarch = None
    stats_r = None
    base = None


class Interface:
    """
    Interface module for Two Time-Scale Transfer Learning (TTSTL) framework.
    
    This module performs model subspace selection and structural break detection
    between the Backbone and Adapter modules.
    """
    
    def __init__(self,
                 backbone_module: Backbone,
                 alpha_enet_grid: Optional[List[float]] = None,
                 lambda_enet_grid: Optional[List[float]] = None,
                 tau_adalasso_grid: Optional[List[float]] = None,
                 lambda_adalasso_grid: Optional[List[float]] = None):
        """
        Initialize the Interface module.
        
        Args:
            backbone_module: The backbone module containing the trained model and covariates
            alpha_grid: Grid of alpha values for elastic net hyperparameter tuning
            lambda_grid: Grid of lambda values for adaptive lasso hyperparameter tuning  
            gamma_grid: Grid of gamma values for adaptive lasso weights
            significance_threshold: Threshold for statistical significance tests
            min_significant_bars: Minimum number of adapter bars where a variable must be significant
            structural_break_threshold: Number of rejections to flag structural break
        """
        self.backbone = backbone_module
        
        # Hyperparameter grids
        self.alpha_enet_grid = alpha_enet_grid
        self.lambda_enet_grid = lambda_enet_grid
        self.tau_adalasso_grid = tau_adalasso_grid
        self.lambda_adalasso_grid = lambda_adalasso_grid

        self.best_alpha_enet = None
        self.best_lambda_enet = None
        self.best_tau_adalasso = None
        self.best_lambda_adalasso = None

        

    def _prepare_data(self, 
                      event_driven_backbone_time_bar_data: pd.DataFrame,
                      physical_time_driven_backbone_time_bar_data: pd.DataFrame, 
                      quantile: float = 0.005,
                      parsimonious_mode: bool = False,
                      target_col_name: str = 'micro_return',
                      n_lags: int = 5) -> pd.DataFrame:
        """
        Prepare data by augmenting with neural covariates, adding adapter bars, 
        including lagged target values, and applying standard scaling.
        
        Args:
            event_driven_backbone_time_bar_data: Event-driven data for neural covariate extraction
            physical_time_driven_backbone_time_bar_data: Physical time-driven data to augment
            quantile: Winsorization quantile for neural covariate extraction
            parsimonious_mode: Whether to use parsimonious mode for neural covariate extraction
            target_col_name: Name of target variable column
            n_lags: Number of lagged values of target variable to include
            
        Returns:
            DataFrame with neural covariates, adapter bars, lagged targets, and standardized variables
        """
        if self.backbone.full_covariate_list is None:
            raise ValueError("Backbone module must have full_covariate_list defined before preparing data.")

        # Augment physical_time_data with neural covariates from backbone
        print("Extracting neural covariates from backbone...")
        neural_covariates = self.backbone._extract_neural_covariates(
            event_driven_data=event_driven_backbone_time_bar_data,
            physical_time_data=physical_time_driven_backbone_time_bar_data,
            quantile=quantile,
            parsimonious_mode=parsimonious_mode
        )

        # Ensure timestamp columns are float type for proper merging
        neural_covariates['timestamp'] = neural_covariates['timestamp'].astype(float)
        physical_time_driven_backbone_time_bar_data['timestamp'] = \
            physical_time_driven_backbone_time_bar_data['timestamp'].astype(float)

        number_of_neural_covariates = len(neural_covariates.columns)

        # Merge neural covariates with physical time data
        augmented_physical_data = pd.merge_asof(
            physical_time_driven_backbone_time_bar_data, 
            neural_covariates, 
            on='timestamp',
            direction='nearest',
            tolerance=1e-6
        )
        
        print(f"Added {number_of_neural_covariates} neural covariates to physical time data")
        
        # Assign adapter time bars using delta_adapter from backbone
        delta_adapter = self.backbone.delta_adapter
        augmented_physical_data['adapter_bar'] = (
            np.floor((augmented_physical_data['timestamp'] + 1e-6) / delta_adapter).astype(int)
        )
        
        # Create lagged values of target variable
        print(f"Creating {n_lags} lagged values of {target_col_name}...")
        lag_columns = []
        for lag in range(1, n_lags + 1):
            lag_col_name = f"{target_col_name}_lag_{lag}"
            augmented_physical_data[lag_col_name] = augmented_physical_data[target_col_name].shift(lag)
            lag_columns.append(lag_col_name)
        
        # Drop rows with NaN values from lagging (first n_lags rows)
        augmented_physical_data = augmented_physical_data.iloc[n_lags:].reset_index(drop=True)
        print(f"Dropped {n_lags} rows due to lagging, remaining data shape: {augmented_physical_data.shape}")
        
        # Prepare extended covariate list including lagged target variables
        extended_covariate_list = self.backbone.full_covariate_list + lag_columns
        
        # Select relevant columns for interface processing
        relevant_columns = ['timestamp', 'adapter_bar'] + extended_covariate_list + [target_col_name]
        augmented_physical_data = augmented_physical_data[relevant_columns].copy()
        
        # Apply standard scaling to all variables (both covariates and target)
        print("Applying standard scaling to all variables...")
        scaler = StandardScaler()
        
        # Scale all variables except timestamp and adapter_bar
        variables_to_scale = extended_covariate_list + [target_col_name]
        augmented_physical_data[variables_to_scale] = scaler.fit_transform(
            augmented_physical_data[variables_to_scale]
        )
        
        # Store the scaler for potential future use
        self.data_scaler = scaler
        self.scaled_variables = variables_to_scale
        self.extended_covariate_list = extended_covariate_list
        
        print(f"Prepared data shape: {augmented_physical_data.shape}")
        print(f"Extended covariates (including {n_lags} lags): {len(extended_covariate_list)}")
        print(f"Columns: {list(augmented_physical_data.columns)}")
        
        return augmented_physical_data

    def _split_data(self, 
                    augmented_data: pd.DataFrame,
                    validation_proportion: float = 0.2,
                    target_col_name: str = 'micro_return') -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
        """
        Split augmented data into training and validation sets by adapter bars.
        
        Args:
            augmented_data: DataFrame with neural covariates and adapter bars
            validation_proportion: Proportion of adapter bars to use for validation
            target_col_name: Name of target variable column
            
        Returns:
            Tuple of (train_data, validation_data, validation_bars)
        """
        # Check required columns
        required_columns = ['timestamp', 'adapter_bar', target_col_name]
        for col in required_columns:
            if col not in augmented_data.columns:
                raise ValueError(f"Required column '{col}' not found in augmented data")
        
        if not hasattr(self, 'extended_covariate_list') or self.extended_covariate_list is None:
            raise ValueError("Extended covariate list not available. Run _prepare_data first.")
        
        for col in self.extended_covariate_list:
            if col not in augmented_data.columns:
                raise ValueError(f"Covariate column '{col}' not found in augmented data")
        
        # Split data by adapter bars for train/validation
        unique_bars = augmented_data['adapter_bar'].unique()
        total_bars = len(unique_bars)
        n_validation_bars = math.ceil(validation_proportion * total_bars)
        
        # Randomly select validation bars
        validation_bars = np.random.choice(unique_bars, size=n_validation_bars, replace=False)
        train_bars = np.setdiff1d(unique_bars, validation_bars)
        
        print(f"Total adapter bars: {total_bars}")
        print(f"Training bars: {len(train_bars)}, Validation bars: {len(validation_bars)}")
        
        # Split data
        train_data = augmented_data[
            augmented_data['adapter_bar'].isin(train_bars)
        ].copy(deep=True)
        
        validation_data = augmented_data[
            augmented_data['adapter_bar'].isin(validation_bars)
        ].copy(deep=True)
        
        print(f"Train data shape: {train_data.shape}")
        print(f"Validation data shape: {validation_data.shape}")
        
        return train_data, validation_data, validation_bars.tolist()

        

    def _fit_enet(self, 
                  X: np.ndarray, 
                  y: np.ndarray, 
                  lambda_enet: float = 0.0,
                  alpha_enet: float = 1.0) -> Dict:
        """
        Fit an Elastic Net model using R's glmnet via R script.
        
        Args:
            X: Feature matrix
            y: Target vector  
            alpha_enet: Elastic net mixing parameter
            lambda_enet: Regularization parameter
            
        Returns:
            Dictionary with fit object and coefficients
        """
        if glmnet is None:
            raise RuntimeError("glmnet R package not available")
        
        try:
            # Source the R script
            r_script_path = os.path.join(os.path.dirname(__file__), 
                                         'r_scripts', 
                                         'run_enet.R')
            ro.r.source(r_script_path)
            
            # Call the R function
            r_result = ro.r.run_enet(X, y, alpha_enet, lambda_enet)
            
            # Extract results
            fit_obj = r_result.rx2('fit')
            coefficients = np.array(r_result.rx2('coefs'))

            return {
                'fit': fit_obj,
                'coefficients': coefficients
            }

        except Exception as e:
            raise RuntimeError(f"Error fitting elastic net: {e}")
    
    def _update_enet_hyperparameters(self, 
                                     physical_time_train_data: pd.DataFrame,
                                     physical_time_validation_data: pd.DataFrame,
                                     target_col_name = 'micro_return') \
                                        -> Tuple[float, float]:
        """
        Find the best elastic net hyperparameters using grid search.
        
        Args:
            physical_time_train_data: Training data DataFrame
            physical_time_validation_data: Validation data DataFrame
            target_col_name: Name of target variable column
            
        Returns:
            Tuple of (best_alpha, best_lambda)
        """
        best_mse = float('inf')
        best_alpha = None
        best_lambda = None
        
        print(f"Testing {len(self.alpha_enet_grid)} alpha values and {len(self.lambda_enet_grid)} lambda values...")
        
        for alpha in self.alpha_enet_grid:
            for lambda_enet in self.lambda_enet_grid:
                # Get the data (already scaled in _prepare_data)
                if not hasattr(self, 'extended_covariate_list') or self.extended_covariate_list is None:
                    raise ValueError("Extended covariate list not available. Run _prepare_data first.")
                
                X_train = physical_time_train_data[self.extended_covariate_list].values
                X_val = physical_time_validation_data[self.extended_covariate_list].values

                y_train = physical_time_train_data[target_col_name].values
                y_val = physical_time_validation_data[target_col_name].values

                # Fit elastic net
                fit_result = self._fit_enet(X_train, y_train, 
                                        alpha_enet=alpha, 
                                        lambda_enet=lambda_enet)

                # Get coefficients from our new format
                coefs = fit_result['coefficients']
                print(f"Coefficients shape: {coefs.shape}")
                
                if len(coefs) == X_val.shape[1] + 1:  
                    predictions = X_val @ coefs[1:] + coefs[0]
                else:  
                    raise ValueError(f"Coefficient mismatch: expected {X_val.shape[1] + 1} but got {len(coefs)}")
                mse = np.mean((predictions - y_val) ** 2)
                
                print(f"Alpha: {alpha:.3f}, Lambda: {lambda_enet:.4f}, MSE: {mse:.6f}")
                
                # Update best parameters if this is better
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = alpha
                    best_lambda = lambda_enet
                    
                # except Exception as e:
                #     print(f"Error with alpha={alpha}, lambda={lambda_enet}: {e}")
                    
        if best_alpha is None or best_lambda is None:
            raise RuntimeError("No valid hyperparameter combination found")
        
        self.best_alpha_enet = best_alpha
        self.best_lambda_enet = best_lambda
            
        print(f"\nBest hyperparameters: alpha={best_alpha:.3f}, lambda={best_lambda:.4f}, MSE={best_mse:.6f}")
        return best_alpha, best_lambda

    def _fit_flexible_adalasso(self, 
                       X: np.ndarray, 
                       y: np.ndarray, 
                       alpha_enet: float,
                       lambda_enet: float,
                       tau_adalasso : float = 1.0,
                       lambda_adalasso: float = 0.0,
                       garch_order: tuple = (1, 1),
                       distribution_model: str = 'sstd') -> Dict:
        """
        Fit an Adaptive Lasso model using R script with GARCH modeling.
        
        Args:
            X: Feature matrix
            y: Target vector
            alpha_enet: Elastic net mixing parameter for initial fit
            lambda_enet: Elastic net regularization parameter for initial fit
            tau_adalasso: Power parameter for adaptive weights
            lambda_adalasso: Adaptive lasso regularization parameter
            garch_order: GARCH order as tuple (p, q)
            distribution_model: Distribution model for GARCH ('sstd', 'norm', etc.)
            
        Returns:
            Dictionary with adaptive lasso fit, GARCH fit, coefficients, and information criteria
        """
        if glmnet is None or rugarch is None:
            raise RuntimeError("glmnet and rugarch R packages not available")
        
        try:
            # Source the R script
            r_script_path = os.path.join(os.path.dirname(__file__), 
                                         'r_scripts', 
                                         'run_adalasso.R')
            ro.r.source(r_script_path)
            
            # Convert garch_order tuple to R vector
            garch_order_r = ro.IntVector(list(garch_order))
            
            # Call the R function
            r_result = ro.r.run_adalasso(
                y, X, alpha_enet, lambda_enet, 
                tau_adalasso, lambda_adalasso,
                garch_order_r, distribution_model
            )
            
            # Extract results
            ada_fit = r_result.rx2('ada_fit')
            garch_fit = r_result.rx2('garch_fit')
            enet_coef = np.array(r_result.rx2('enet_coef'))
            penalty_weights = np.array(r_result.rx2('penalty_weights'))
            ada_coef = np.array(r_result.rx2('ada_coef'))
            aic = float(r_result.rx2('AIC')[0])
            bic = float(r_result.rx2('BIC')[0])
            
            return {
                'ada_fit': ada_fit,
                'garch_fit': garch_fit,
                'enet_coefficients': enet_coef,
                'penalty_weights': penalty_weights,
                'ada_coefficients': ada_coef,
                'AIC': aic,
                'BIC': bic,
                'alpha_enet': alpha_enet,
                'lambda_enet': lambda_enet,
                'tau_adalasso': tau_adalasso,
                'lambda_adalasso': lambda_adalasso
            }
            
        except Exception as e:
            raise RuntimeError(f"Error fitting adaptive lasso: {e}")
        
    def _update_ada_lasso_hyperparameters(self,
                                          physical_time_driven_backbone_time_bar_data: pd.DataFrame,
                                          alpha_enet: float = 1.0,
                                          lambda_enet: float = 0.0,
                                          target_col_name: str = 'micro_return') -> Dict:
        """
        Find the best adaptive lasso hyperparameters using BIC across adapter time bars.
        
        Args:
            physical_time_driven_backbone_time_bar_data: Prepared data with neural covariates and adapter bars
            alpha_enet: Elastic net mixing parameter for initial fit
            lambda_enet: Elastic net regularization parameter for initial fit
            target_col_name: Name of target variable column
            
        Returns:
            Dictionary with best hyperparameters and BIC results
        """
        if self.tau_adalasso_grid is None or self.lambda_adalasso_grid is None:
            raise ValueError("tau_adalasso_grid and lambda_adalasso_grid must be defined")
        
        if 'adapter_bar' not in physical_time_driven_backbone_time_bar_data.columns:
            raise ValueError("Input data must contain 'adapter_bar' column. Use _prepare_data first.")
        
        best_bic = float('inf')
        best_tau = None
        best_lambda = None
        results = []
        
        print(f"Testing {len(self.tau_adalasso_grid)} tau values and {len(self.lambda_adalasso_grid)} lambda values...")
        print(f"Number of adapter bars: {physical_time_driven_backbone_time_bar_data['adapter_bar'].nunique()}")
        
        for tau_adalasso in self.tau_adalasso_grid:
            for lambda_adalasso in self.lambda_adalasso_grid:
                total_bic = 0.0
                valid_bars = 0
                
                print(f"Testing tau={tau_adalasso:.3f}, lambda={lambda_adalasso:.6f}")
                
                # Iterate through each adapter time bar
                for adapter_bar in physical_time_driven_backbone_time_bar_data['adapter_bar'].unique():
                    this_adapter_bar_data = physical_time_driven_backbone_time_bar_data[
                        physical_time_driven_backbone_time_bar_data['adapter_bar'] == adapter_bar
                    ]
                    
                    # Skip bars with insufficient data
                    if len(this_adapter_bar_data) < 10:
                        continue
                    
                    try:
                        # Get features and target for this bar (already scaled in _prepare_data)
                        if not hasattr(self, 'extended_covariate_list') or self.extended_covariate_list is None:
                            raise ValueError("Extended covariate list not available. Run _prepare_data first.")
                        
                        X_adapter_bar = this_adapter_bar_data[self.extended_covariate_list].values
                        y_adapter_bar = this_adapter_bar_data[target_col_name].values
                        
                        # Fit adaptive lasso for this bar
                        ada_result = self._fit_flexible_adalasso(
                            X=X_adapter_bar,
                            y=y_adapter_bar,
                            alpha_enet=alpha_enet,
                            lambda_enet=lambda_enet,
                            tau_adalasso=tau_adalasso,
                            lambda_adalasso=lambda_adalasso
                        )
                        
                        # Add BIC from this bar
                        adapter_bar_bic = ada_result['BIC']
                        total_bic += adapter_bar_bic
                        valid_bars += 1
                        
                    except Exception as e:
                        print(f"  Warning: Error processing adapter bar {adapter_bar}: {e}")
                        continue
                
                # Skip this hyperparameter combination if no valid bars
                if valid_bars == 0:
                    print(f"  No valid bars for tau={tau_adalasso:.3f}, lambda={lambda_adalasso:.6f}")
                    continue
                
                # Compute average BIC across all adapter bars
                average_bic = total_bic / valid_bars
                
                results.append({
                    'tau_adalasso': tau_adalasso,
                    'lambda_adalasso': lambda_adalasso,
                    'average_bic': average_bic,
                    'valid_bars': valid_bars
                })
                
                print(f"  Average BIC: {average_bic:.6f} (from {valid_bars} bars)")
                
                # Update best parameters if this is better
                if average_bic < best_bic:
                    best_bic = average_bic
                    best_tau = tau_adalasso
                    best_lambda = lambda_adalasso
        
        if best_tau is None or best_lambda is None:
            raise RuntimeError("No valid hyperparameter combination found")
        
        # Set as class attributes
        self.best_tau_adalasso = best_tau
        self.best_lambda_adalasso = best_lambda
        
        print(f"\nBest hyperparameters: tau={best_tau:.3f}, lambda={best_lambda:.6f}, Ave. BIC={best_bic:.6f}")
        
        return {
            'best_tau_adalasso': best_tau,
            'best_lambda_adalasso': best_lambda,
            'best_bic': best_bic,
            'all_results': results,
            'alpha_enet': alpha_enet,
            'lambda_enet': lambda_enet
        }

    def _update_hyperparameters(self, 
                                 event_driven_backbone_time_bar_data: pd.DataFrame,
                                 physical_time_driven_backbone_time_bar_data: pd.DataFrame,
                                 quantile: float = 0.005,
                                 parsimonious_mode: bool = False,
                                 target_col_name: str = 'micro_return',
                                 n_lags: int = 5,
                                 validation_proportion: float = 0.2) -> Dict:
        """
        Complete hyperparameter tuning procedure for both Elastic Net and Adaptive Lasso.
        
        This function implements the full hyperparameter tuning workflow:
        1. Prepare data with neural covariates, adapter bars, lagged targets, and scaling
        2. Split data into training and validation sets
        3. Tune Elastic Net hyperparameters (alpha, lambda) using MSE on validation set
        4. Keeping Elastic Net parameters fixed, tune Adaptive Lasso hyperparameters (tau, lambda) using BIC
        
        Args:
            event_driven_backbone_time_bar_data: Event-driven data for neural covariate extraction
            physical_time_driven_backbone_time_bar_data: Physical time-driven data to augment
            quantile: Winsorization quantile for neural covariate extraction
            parsimonious_mode: Whether to use parsimonious mode for neural covariate extraction
            target_col_name: Name of target variable column
            n_lags: Number of lagged values of target variable to include
            validation_proportion: Proportion of adapter bars to use for validation
            
        Returns:
            Dictionary with all hyperparameter tuning results
        """
        print("\n=== Starting Complete Hyperparameter Tuning ===")
        
        # Step 1: Prepare data with neural covariates, adapter bars, lagged targets, and scaling
        print("Preparing data...")
        augmented_data = self._prepare_data(
            event_driven_backbone_time_bar_data=event_driven_backbone_time_bar_data,
            physical_time_driven_backbone_time_bar_data=physical_time_driven_backbone_time_bar_data,
            quantile=quantile,
            parsimonious_mode=parsimonious_mode,
            target_col_name=target_col_name,
            n_lags=n_lags
        )
        
        # Step 2: Split data into training and validation sets by adapter bars
        print("Splitting data into training and validation sets...")
        train_data, validation_data, validation_bars = self._split_data(
            augmented_data=augmented_data,
            validation_proportion=validation_proportion,
            target_col_name=target_col_name
        )
        
        # Step 3: Tune Elastic Net hyperparameters using MSE on validation set
        print("Tuning Elastic Net hyperparameters...")
        if self.alpha_enet_grid is None or self.lambda_enet_grid is None:
            raise ValueError("alpha_enet_grid and lambda_enet_grid must be defined for hyperparameter tuning")
        
        best_alpha_enet, best_lambda_enet = self._update_enet_hyperparameters(
            physical_time_train_data=train_data,
            physical_time_validation_data=validation_data,
            target_col_name=target_col_name
        )
        
        # Step 4: Tune Adaptive Lasso hyperparameters using BIC, keeping Elastic Net parameters fixed
        print("Tuning Adaptive Lasso hyperparameters with fixed Elastic Net parameters...")
        ada_lasso_results = self._update_ada_lasso_hyperparameters(
            physical_time_driven_backbone_time_bar_data=augmented_data,
            alpha_enet=best_alpha_enet,
            lambda_enet=best_lambda_enet,
            target_col_name=target_col_name
        )
        
        # Compile complete results
        hyperparameter_results = {
            'elastic_net': {
                'best_alpha': best_alpha_enet,
                'best_lambda': best_lambda_enet,
                'alpha_grid': self.alpha_enet_grid,
                'lambda_grid': self.lambda_enet_grid
            },
            'adaptive_lasso': {
                'best_tau': ada_lasso_results['best_tau_adalasso'],
                'best_lambda': ada_lasso_results['best_lambda_adalasso'],
                'best_bic': ada_lasso_results['best_bic'],
                'tau_grid': self.tau_adalasso_grid,
                'lambda_grid': self.lambda_adalasso_grid,
                'all_results': ada_lasso_results['all_results']
            },
            'data_info': {
                'total_samples': len(augmented_data),
                'training_samples': len(train_data),
                'validation_samples': len(validation_data),
                'n_features': len(self.extended_covariate_list),
                'n_lags': n_lags,
                'validation_bars': validation_bars
            }
        }
        
        print(f"\n=== Hyperparameter Tuning Complete ===")
        print(f"Best Elastic Net: alpha={best_alpha_enet:.3f}, lambda={best_lambda_enet:.4f}")
        print(f"Best Adaptive Lasso: tau={ada_lasso_results['best_tau_adalasso']:.3f}, lambda={ada_lasso_results['best_lambda_adalasso']:.6f}")
        print(f"Features used: {len(self.extended_covariate_list)} (including {n_lags} lags)")
        
        return hyperparameter_results

    def _fit_ols_HC(self, X: np.ndarray, y: np.ndarray) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Fit OLS model with heteroskedasticity-robust standard errors.
        
        Args:
            X: Feature matrix (should include constant if needed)
            y: Target vector
            
        Returns:
            Fitted OLS model with robust standard errors
        """
        try:
            # Fit OLS model with heteroskedasticity-robust standard errors
            model = OLS(y, X).fit(cov_type='HC0')
            return model

        except Exception as e:
            warnings.warn(f"Error in OLS fitting: {e}")
            return None
        

    def global_update(self, 
                      event_driven_backbone_time_bar_data: pd.DataFrame,
                      physical_time_driven_backbone_time_bar_data: pd.DataFrame,
                      quantile: float = 0.005,
                      parsimonious_mode: bool = False,
                      target_col_name: str = 'micro_return',
                      n_lags: int = 5,
                      validation_proportion: float = 0.2,
                      selection_threshold: float = 0.5,
                      tolerance_threshold: float = 1e-6,
                      ols_significance_level: float = 0.05) -> Dict:
        """
        Perform global variable selection for the TTSTL Interface module.
        
        This method implements the complete global variable selection workflow:
        1. Run hyperparameter tuning to find optimal Elastic Net and Adaptive Lasso parameters
        2. Apply Flexible Adaptive Lasso to each adapter time bar using optimal hyperparameters
        3. Extract coefficients above tolerance threshold from each adapter bar
        4. Apply OLS with heteroskedasticity-robust tests to significant coefficients
        5. Select variables that appear as significant across multiple adapter bars
        
        Args:
            event_driven_backbone_time_bar_data: Event-driven data for neural covariate extraction
            physical_time_driven_backbone_time_bar_data: Physical time-driven data to augment
            quantile: Winsorization quantile for neural covariate extraction
            parsimonious_mode: Whether to use parsimonious mode for neural covariate extraction
            target_col_name: Name of target variable column
            n_lags: Number of lagged values of target variable to include
            validation_proportion: Proportion of adapter bars to use for validation in hyperparameter tuning
            selection_threshold: Minimum proportion of adapter bars where variable must be significant
            tolerance_threshold: Minimum absolute coefficient value to consider in Adaptive Lasso
            ols_significance_level: Significance level for OLS heteroskedasticity-robust tests
            
        Returns:
            Dictionary with selected variables and global update results
        """
        print("\n" + "="*60)
        print("STARTING INTERFACE GLOBAL UPDATE")
        print("="*60)
        
        # Step 1: Run hyperparameter tuning
        print("Step 1: Running hyperparameter tuning...")
        hyperparameter_results = self._update_hyperparameters(
            event_driven_backbone_time_bar_data=event_driven_backbone_time_bar_data,
            physical_time_driven_backbone_time_bar_data=physical_time_driven_backbone_time_bar_data,
            quantile=quantile,
            parsimonious_mode=parsimonious_mode,
            target_col_name=target_col_name,
            n_lags=n_lags,
            validation_proportion=validation_proportion
        )
        
        # Extract optimal hyperparameters from class attributes (set by _update_hyperparameters)
        best_alpha_enet = self.best_alpha_enet
        best_lambda_enet = self.best_lambda_enet
        best_tau_adalasso = self.best_tau_adalasso
        best_lambda_adalasso = self.best_lambda_adalasso
        
        print(f"Optimal hyperparameters found:")
        print(f"  Elastic Net: α={best_alpha_enet:.3f}, λ={best_lambda_enet:.4f}")
        print(f"  Adaptive Lasso: τ={best_tau_adalasso:.3f}, λ={best_lambda_adalasso:.6f}")
        
        # Get the prepared data from hyperparameter tuning (it should be stored in the interface)
        # We need to re-prepare the data since _update_hyperparameters used a subset for validation
        print("Step 2: Re-preparing full dataset for global variable selection...")
        augmented_data = self._prepare_data(
            event_driven_backbone_time_bar_data=event_driven_backbone_time_bar_data,
            physical_time_driven_backbone_time_bar_data=physical_time_driven_backbone_time_bar_data,
            quantile=quantile,
            parsimonious_mode=parsimonious_mode,
            target_col_name=target_col_name,
            n_lags=n_lags
        )
        
        # Get unique adapter bars
        unique_adapter_bars = augmented_data['adapter_bar'].unique()
        n_adapter_bars = len(unique_adapter_bars)
        print(f"Processing {n_adapter_bars} adapter time bars for variable selection...")
        
        # Initialize variable selection tracking
        variable_significance_count = {}
        for var in self.extended_covariate_list:
            variable_significance_count[var] = 0
        
        processed_bars = 0
        failed_bars = 0
        
        # Step 3: Process each adapter time bar
        print("Step 3: Processing each adapter time bar with Flexible Adaptive Lasso + OLS...")
        
        for i, adapter_bar in enumerate(unique_adapter_bars):
            print(f"Processing adapter bar {adapter_bar} ({i+1}/{n_adapter_bars})...")
            
            # Extract data for this adapter bar
            bar_data = augmented_data[
                augmented_data['adapter_bar'] == adapter_bar
            ].copy()
            
            # Skip bars with insufficient data
            if len(bar_data) < 10:
                print(f"  Skipping adapter bar {adapter_bar}: insufficient data ({len(bar_data)} samples)")
                failed_bars += 1
                continue
            
            try:
                # Get features and target for this adapter bar
                X_bar = bar_data[self.extended_covariate_list].values
                y_bar = bar_data[target_col_name].values
                
                # Step 3a: Apply Flexible Adaptive Lasso with optimal hyperparameters
                ada_result = self._fit_flexible_adalasso(
                    X=X_bar,
                    y=y_bar,
                    alpha_enet=best_alpha_enet,
                    lambda_enet=best_lambda_enet,
                    tau_adalasso=best_tau_adalasso,
                    lambda_adalasso=best_lambda_adalasso
                )
                
                # Step 3b: Extract coefficients above tolerance threshold
                ada_coefficients = ada_result['ada_coefficients']
                
                # Find variables with coefficients above tolerance threshold
                # Note: ada_coefficients may include intercept, so we need to handle indexing
                if len(ada_coefficients) == len(self.extended_covariate_list) + 1:
                    # Coefficients include intercept
                    significant_coef_mask = np.abs(ada_coefficients[1:]) > tolerance_threshold
                    intercept = ada_coefficients[0]
                    coef_without_intercept = ada_coefficients[1:]
                elif len(ada_coefficients) == len(self.extended_covariate_list):
                    # Coefficients don't include intercept
                    significant_coef_mask = np.abs(ada_coefficients) > tolerance_threshold
                    intercept = 0.0
                    coef_without_intercept = ada_coefficients
                else:
                    raise ValueError(f"Coefficient dimension mismatch: expected {len(self.extended_covariate_list)} or {len(self.extended_covariate_list)+1}, got {len(ada_coefficients)}")
                
                # Get indices of variables with significant coefficients
                significant_var_indices = np.where(significant_coef_mask)[0]
                
                if len(significant_var_indices) == 0:
                    print(f"  No variables above tolerance threshold for adapter bar {adapter_bar}")
                    processed_bars += 1
                    continue
                
                print(f"  {len(significant_var_indices)} variables above tolerance threshold")
                
                # Step 3c: Apply OLS with heteroskedasticity-robust tests
                # Select only the variables with significant coefficients
                X_selected = X_bar[:, significant_var_indices]
                
                # Add constant for OLS
                X_selected_with_const = sm.add_constant(X_selected)
                
                # Fit OLS with heteroskedasticity-robust standard errors
                ols_model = self._fit_ols_HC(
                    X=X_selected_with_const, 
                    y=y_bar
                )
                
                if ols_model is not None:
                    # Get p-values (skip intercept)
                    p_values = ols_model.pvalues[1:]
                    
                    # Find significant variables
                    ols_significant_mask = p_values < ols_significance_level
                    
                    # Map back to original variable indices
                    ols_significant_indices = significant_var_indices[ols_significant_mask]
                    
                    # Update significance counts for OLS-significant variables
                    for idx in ols_significant_indices:
                        var_name = self.extended_covariate_list[idx]
                        variable_significance_count[var_name] += 1
                    
                    print(f"  {len(ols_significant_indices)} variables significant in OLS test")
                else:
                    print(f"  OLS test failed for adapter bar {adapter_bar}")
                
                processed_bars += 1
                
            except Exception as e:
                print(f"  Error processing adapter bar {adapter_bar}: {e}")
                failed_bars += 1
                continue
        
        # Step 4: Select variables based on selection threshold
        print(f"\nStep 4: Selecting variables based on selection threshold...")
        print(f"Successfully processed: {processed_bars} bars")
        print(f"Failed to process: {failed_bars} bars")
        
        if processed_bars == 0:
            raise RuntimeError("No adapter bars were successfully processed")
        
        # Calculate minimum number of bars for selection
        min_bars_for_selection = max(1, int(selection_threshold * processed_bars))
        print(f"Selection threshold: {selection_threshold:.2%} = minimum {min_bars_for_selection} bars")
        
        # Select variables that appear in at least the minimum number of bars
        selected_variables = []
        variable_selection_info = {}
        
        for var_name, count in variable_significance_count.items():
            proportion = count / processed_bars if processed_bars > 0 else 0
            variable_selection_info[var_name] = {
                'count': count,
                'proportion': proportion,
                'selected': count >= min_bars_for_selection
            }
            
            if count >= min_bars_for_selection:
                selected_variables.append(var_name)
        
        # Store selected variables and parameters in the interface
        self.selected_variables = selected_variables
        self.tolerance_threshold = tolerance_threshold
        
        # Sort variables by selection frequency for reporting
        sorted_vars = sorted(variable_selection_info.items(), 
                           key=lambda x: x[1]['count'], reverse=True)
        
        print(f"\n" + "="*50)
        print(f"GLOBAL VARIABLE SELECTION RESULTS")
        print(f"="*50)
        print(f"Selected {len(selected_variables)} variables out of {len(self.extended_covariate_list)}")
        print(f"Selection criteria: significant in ≥{min_bars_for_selection} bars ({selection_threshold:.1%})")
        
        print(f"\nSelected Variables:")
        for i, var in enumerate(selected_variables, 1):
            info = variable_selection_info[var]
            print(f"  {i:2d}. {var:<30} ({info['count']:2d}/{processed_bars} bars, {info['proportion']:.1%})")
        
        print(f"\nTop 10 Variables by Selection Frequency:")
        for i, (var, info) in enumerate(sorted_vars[:10], 1):
            status = "✓" if info['selected'] else "✗"
            print(f"  {i:2d}. {status} {var:<30} ({info['count']:2d}/{processed_bars} bars, {info['proportion']:.1%})")
        
        # Compile results
        global_update_results = {
            'selected_variables': selected_variables,
            'n_selected': len(selected_variables),
            'n_total_variables': len(self.extended_covariate_list),
            'selection_threshold': selection_threshold,
            'tolerance_threshold': tolerance_threshold,
            'ols_significance_level': ols_significance_level,
            'hyperparameter_results': hyperparameter_results,
            'processing_stats': {
                'total_adapter_bars': n_adapter_bars,
                'processed_bars': processed_bars,
                'failed_bars': failed_bars,
                'min_bars_for_selection': min_bars_for_selection
            },
            'variable_selection_info': variable_selection_info,
            'extended_covariate_list': self.extended_covariate_list
        }
        
        print(f"\nGlobal Update Complete!")
        print(f"{len(selected_variables)} variables selected for adapter-level modeling")
        print("="*60)
        
        return global_update_results

    def local_adaptation(self, 
                         adapter_bar_data: pd.DataFrame,
                         target_col_name: str = 'micro_return',
                         significance_level: float = 0.05) -> Dict:
        """
        Perform local adaptation for a specific adapter time bar using Wald test.
        
        This method implements the local adaptation procedure from section 7.3.2:
        1. Start with globally selected variables from global_update
        2. Run Flexible Adaptive Lasso on the adapter bar to get new variable selection
        3. Identify newly selected variables (set difference)
        4. Use Wald test to check if the newly selected variables are jointly significant
        5. Return results with Wald test statistics
        
        Args:
            adapter_bar_data: Data for specific adapter time bar (already filtered)
            target_col_name: Name of target variable column
            significance_level: Significance level for Wald test
            
        Returns:
            Dictionary with local adaptation results including Wald test statistics
        """
        if not hasattr(self, 'selected_variables') or self.selected_variables is None:
            raise ValueError("Global update must be run first to establish global variable selection")
        
        if not hasattr(self, 'extended_covariate_list') or self.extended_covariate_list is None:
            raise ValueError("Extended covariate list not available. Run global_update first.")
        
        print("\n" + "="*60)
        print("STARTING LOCAL ADAPTATION")
        print("="*60)
        print(f"Adapter bar data shape: {adapter_bar_data.shape}")
        print(f"Global variables: {len(self.selected_variables)}")
        
        # Check required attributes from global update
        required_attrs = ['best_alpha_enet', 'best_lambda_enet', 'best_tau_adalasso', 'best_lambda_adalasso']
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise ValueError(f"Missing required attribute {attr}. Run global_update first.")
        
        try:
            # Step 1: Extract features and target for this adapter bar
            X_full = adapter_bar_data[self.extended_covariate_list].values
            y = adapter_bar_data[target_col_name].values
            
            print(f"Full feature matrix shape: {X_full.shape}")
            print(f"Target vector shape: {y.shape}")
            
            # Step 2: Apply Flexible Adaptive Lasso with globally optimal hyperparameters
            print("Step 1: Running Flexible Adaptive Lasso for local variable selection...")
            ada_result = self._fit_flexible_adalasso(
                X=X_full,
                y=y,
                alpha_enet=self.best_alpha_enet,
                lambda_enet=self.best_lambda_enet,
                tau_adalasso=self.best_tau_adalasso,
                lambda_adalasso=self.best_lambda_adalasso
            )
            
            # Step 3: Extract locally selected variables
            ada_coefficients = ada_result['ada_coefficients']
            
            # Handle coefficient indexing (with or without intercept)
            if len(ada_coefficients) == len(self.extended_covariate_list) + 1:
                # Coefficients include intercept
                coef_without_intercept = ada_coefficients[1:]
            elif len(ada_coefficients) == len(self.extended_covariate_list):
                # Coefficients don't include intercept
                coef_without_intercept = ada_coefficients
            else:
                raise ValueError(f"Coefficient dimension mismatch: expected {len(self.extended_covariate_list)} or {len(self.extended_covariate_list)+1}, got {len(ada_coefficients)}")
            
            # Find locally selected variables (non-zero coefficients)
            tolerance_threshold = getattr(self, 'tolerance_threshold', 1e-6)
            local_selection_mask = np.abs(coef_without_intercept) > tolerance_threshold
            local_selected_indices = np.where(local_selection_mask)[0]
            local_selected_variables = [self.extended_covariate_list[i] for i in local_selected_indices]
            
            print(f"Local Adaptive Lasso selected: {len(local_selected_variables)} variables")
            
            # Step 4: Identify newly selected variables (set difference)
            global_var_set = set(self.selected_variables)
            local_var_set = set(local_selected_variables)
            newly_selected_variables = list(local_var_set - global_var_set)
            
            print(f"Global variables: {len(self.selected_variables)}")
            print(f"Newly selected variables: {len(newly_selected_variables)}")
            
            if len(newly_selected_variables) == 0:
                print("No new variables selected locally. Using global selection.")
                final_variables = self.selected_variables.copy()
                wald_test_result = None
            else:
                print(f"Testing {len(newly_selected_variables)} newly selected variables:")
                for var in newly_selected_variables:
                    print(f"  - {var}")
                
                # Step 5: Perform Wald test
                print("Step 2: Performing Wald test for newly selected variables...")
                wald_test_result = self._wald_test_new_variables(
                    X_full=X_full,
                    y=y,
                    global_variables=self.selected_variables,
                    newly_selected_variables=newly_selected_variables,
                    significance_level=significance_level
                )
                
                # Step 6: Determine final variable set based on Wald test
                if wald_test_result['is_significant']:
                    final_variables = self.selected_variables + newly_selected_variables
                    print(f"✓ Wald test significant (p={wald_test_result['p_value']:.6f})")
                    print(f"  Including {len(newly_selected_variables)} new variables in final model")
                else:
                    final_variables = self.selected_variables.copy()
                    print(f"✗ Wald test not significant (p={wald_test_result['p_value']:.6f})")
                    print(f"  Using only global variables in final model")
            
            # Compile results
            local_adaptation_results = {
                'adapter_bar_data_shape': adapter_bar_data.shape,
                'global_variables': self.selected_variables,
                'local_selected_variables': local_selected_variables,
                'newly_selected_variables': newly_selected_variables,
                'final_variables': final_variables,
                'n_global': len(self.selected_variables),
                'n_local': len(local_selected_variables),
                'n_newly_selected': len(newly_selected_variables),
                'n_final': len(final_variables),
                'wald_test_result': wald_test_result,
                'adaptive_lasso_result': ada_result,
                'hyperparameters': {
                    'alpha_enet': self.best_alpha_enet,
                    'lambda_enet': self.best_lambda_enet,
                    'tau_adalasso': self.best_tau_adalasso,
                    'lambda_adalasso': self.best_lambda_adalasso
                }
            }
            
            print(f"\n" + "="*50)
            print(f"LOCAL ADAPTATION RESULTS")
            print(f"="*50)
            print(f"Global variables: {len(self.selected_variables)}")
            print(f"Local adaptive lasso selected: {len(local_selected_variables)}")
            print(f"Newly selected: {len(newly_selected_variables)}")
            print(f"Final model variables: {len(final_variables)}")
            if wald_test_result:
                print(f"Wald test p-value: {wald_test_result['p_value']:.6f}")
                print(f"Wald test significant: {wald_test_result['is_significant']}")
            print("="*50)
            
            return local_adaptation_results
            
        except Exception as e:
            raise RuntimeError(f"Error in local adaptation: {e}")

    def _wald_test_new_variables(self, 
                                 X_full: np.ndarray, 
                                 y: np.ndarray,
                                 global_variables: List[str],
                                 newly_selected_variables: List[str],
                                 significance_level: float = 0.05) -> Dict:
        """
        Perform Wald test to check if newly selected variables are jointly significant.
        
        Args:
            X_full: Full feature matrix
            y: Target vector
            global_variables: List of globally selected variable names
            newly_selected_variables: List of newly selected variable names
            significance_level: Significance level for the test
            
        Returns:
            Dictionary with Wald test results
        """
        try:
            # Get indices for global and newly selected variables
            global_indices = [self.extended_covariate_list.index(var) for var in global_variables]
            new_indices = [self.extended_covariate_list.index(var) for var in newly_selected_variables]
            all_indices = global_indices + new_indices
            
            # Create design matrices
            X_restricted = X_full[:, global_indices]  # Model with only global variables
            X_unrestricted = X_full[:, all_indices]   # Model with global + new variables
            
            # Add constants
            X_restricted_const = sm.add_constant(X_restricted)
            X_unrestricted_const = sm.add_constant(X_unrestricted)
            
            # Fit both models
            model_restricted = self._fit_ols_HC(X=X_restricted_const, y=y)
            model_unrestricted = self._fit_ols_HC(X=X_unrestricted_const, y=y)
            
            if model_restricted is None or model_unrestricted is None:
                raise RuntimeError("Failed to fit models for Wald test")
            
            # Perform Wald test
            # Test H0: coefficients of newly selected variables = 0
            # The test statistic is based on the coefficient vector and covariance matrix
            
            # Get coefficients and covariance matrix for newly selected variables
            # Note: coefficients include intercept, so we need to account for that
            n_global_with_const = len(global_indices) + 1  # +1 for intercept
            
            # Extract coefficients for newly selected variables
            new_var_coefs = model_unrestricted.params[n_global_with_const:]
            
            # Extract covariance matrix for newly selected variables  
            # Convert to numpy array if it's a pandas DataFrame
            cov_params = model_unrestricted.cov_params()
            if hasattr(cov_params, 'iloc'):
                new_var_cov = cov_params.iloc[n_global_with_const:, n_global_with_const:].values
            else:
                new_var_cov = cov_params[n_global_with_const:, n_global_with_const:]
            
            # Convert coefficients to numpy array if needed
            if hasattr(new_var_coefs, 'values'):
                new_var_coefs = new_var_coefs.values
            
            # Compute Wald statistic: coef' * inv(cov) * coef
            try:
                wald_statistic = float(new_var_coefs.T @ np.linalg.inv(new_var_cov) @ new_var_coefs)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if matrix is singular
                wald_statistic = float(new_var_coefs.T @ np.linalg.pinv(new_var_cov) @ new_var_coefs)
            
            # Degrees of freedom = number of newly selected variables
            df = len(newly_selected_variables)
            
            # P-value from chi-squared distribution
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(wald_statistic, df)
            
            is_significant = p_value < significance_level
            
            return {
                'wald_statistic': wald_statistic,
                'degrees_of_freedom': df,
                'p_value': p_value,
                'significance_level': significance_level,
                'is_significant': is_significant,
                'newly_selected_variables': newly_selected_variables,
                'model_restricted': model_restricted,
                'model_unrestricted': model_unrestricted,
                'restricted_aic': model_restricted.aic,
                'unrestricted_aic': model_unrestricted.aic,
                'restricted_bic': model_restricted.bic,
                'unrestricted_bic': model_unrestricted.bic
            }
            
        except Exception as e:
            raise RuntimeError(f"Error in Wald test: {e}")
    

