import os
import math
import warnings
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.numpy2ri as numpy2ri

from sklearn.preprocessing import StandardScaler

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
        print("Step 1: Preparing data...")
        augmented_data = self._prepare_data(
            event_driven_backbone_time_bar_data=event_driven_backbone_time_bar_data,
            physical_time_driven_backbone_time_bar_data=physical_time_driven_backbone_time_bar_data,
            quantile=quantile,
            parsimonious_mode=parsimonious_mode,
            target_col_name=target_col_name,
            n_lags=n_lags
        )
        
        # Step 2: Split data into training and validation sets by adapter bars
        print("Step 2: Splitting data into training and validation sets...")
        train_data, validation_data, validation_bars = self._split_data(
            augmented_data=augmented_data,
            validation_proportion=validation_proportion,
            target_col_name=target_col_name
        )
        
        # Step 3: Tune Elastic Net hyperparameters using MSE on validation set
        print("Step 3: Tuning Elastic Net hyperparameters...")
        if self.alpha_enet_grid is None or self.lambda_enet_grid is None:
            raise ValueError("alpha_enet_grid and lambda_enet_grid must be defined for hyperparameter tuning")
        
        best_alpha_enet, best_lambda_enet = self._update_enet_hyperparameters(
            physical_time_train_data=train_data,
            physical_time_validation_data=validation_data,
            target_col_name=target_col_name
        )
        
        # Step 4: Tune Adaptive Lasso hyperparameters using BIC, keeping Elastic Net parameters fixed
        print("Step 4: Tuning Adaptive Lasso hyperparameters with fixed Elastic Net parameters...")
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

    def _ols_screening(self, X: np.ndarray, y: np.ndarray, 
                      selected_variables: np.ndarray, 
                      feature_names: List[str]) -> List[str]:
        """
        Perform OLS screening with heteroskedasticity-robust t-tests.
        
        Args:
            X: Full feature matrix
            y: Target vector
            selected_variables: Boolean mask of variables selected by adaptive lasso
            feature_names: Names of all features
            
        Returns:
            List of significantly selected feature names
        """
        try:
            from statsmodels.regression.linear_model import OLS
            from statsmodels.stats.diagnostic import het_white
            import statsmodels.api as sm
            
            X_selected = X[:, selected_variables]
            X_with_const = sm.add_constant(X_selected)
            
            # Fit OLS model
            model = OLS(y, X_with_const).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
            
            # Get t-statistics (skip intercept)
            t_stats = model.tvalues[1:]
            p_values = model.pvalues[1:]
            
            # Find significant variables
            significant_mask = p_values < self.significance_threshold
            selected_features = np.array(feature_names)[selected_variables]
            final_selected = selected_features[significant_mask].tolist()
            
            return final_selected
            
        except Exception as e:
            warnings.warn(f"Error in OLS screening: {e}")
            # Fallback: return all selected variables
            return np.array(feature_names)[selected_variables].tolist()

    def global_update(self, augmented_physical_data: pd.DataFrame) -> Dict:
        """
        Conduct hyperparameter tuning and variable selection for global update.
        
        This implements the global update procedure:
        1. Hyperparameter tuning for Elastic Net and Adaptive Lasso
        2. Variable selection using Flexible adaLASSO
        3. OLS screening with heteroskedasticity-robust tests
        4. Selection of variables significant across multiple adapter time bars
        
        Args:
            augmented_physical_data: DataFrame with physical time data including neural covariates
            
        Returns:
            Dictionary with selected variables and model information
        """
        print("\n--- Starting Interface Global Update ---")
        
        # Create adapter time bars
        try:
            augmented_physical_data['adapter_bar'] = (
                np.floor((augmented_physical_data['timestamp'] + 1e-3) / self.backbone.delta_adapter)
                .astype(int)
            )
        except KeyError:
            augmented_physical_data['adapter_bar'] = (
                np.floor((augmented_physical_data.index.to_series() + 1e-3) / self.backbone.delta_adapter)
                .astype(int)
            )
        
        # Prepare data
        X, y, feature_names = self._prepare_data(augmented_physical_data)
        
        print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Step 1: Hyperparameter tuning on full dataset
        print("Step 1: Hyperparameter tuning...")
        self.best_hyperparameters = self._hyperparameter_tuning(X, y)
        
        # Step 2-4: Apply best hyperparameters and get selected variables
        print("Step 2-4: Variable selection with best hyperparameters...")
        enet_result = self._fit_enet(X, y, alpha=self.best_hyperparameters['alpha'])
        ada_result = self._fit_ada_lasso(
            X, y, enet_result['coefficients'],
            lambda_ada=self.best_hyperparameters['lambda_ada'],
            gamma=self.best_hyperparameters['gamma']
        )
        
        # Step 5-6: OLS screening on each adapter time bar
        print("Step 5-6: OLS screening across adapter time bars...")
        significant_vars_per_bar = {}
        
        for bar_id, bar_data in augmented_physical_data.groupby('adapter_bar'):
            if len(bar_data) < 10:  # Skip bars with too few observations
                continue
                
            try:
                X_bar, y_bar, _ = self._prepare_data(bar_data)
                
                # Apply OLS screening with variables selected by adaLASSO
                significant_vars = self._ols_screening(
                    X_bar, y_bar, ada_result['selected_variables'], feature_names
                )
                significant_vars_per_bar[bar_id] = significant_vars
                
            except Exception as e:
                warnings.warn(f"Error processing adapter bar {bar_id}: {e}")
                continue
        
        # Step 7: Select variables significant in minimum number of bars
        print("Step 7: Final variable selection...")
        all_vars = set()
        for vars_list in significant_vars_per_bar.values():
            all_vars.update(vars_list)
        
        var_counts = {}
        for var in all_vars:
            count = sum(1 for vars_list in significant_vars_per_bar.values() if var in vars_list)
            var_counts[var] = count
        
        # Select variables significant in at least min_significant_bars
        min_bars = min(self.min_significant_bars, len(significant_vars_per_bar))
        self.selected_variables = [
            var for var, count in var_counts.items() 
            if count >= min_bars
        ]
        
        print(f"Selected {len(self.selected_variables)} variables out of {len(feature_names)}")
        print(f"Selected variables: {self.selected_variables}")
        
        # Store global model coefficients for structural break detection
        if self.selected_variables:
            selected_idx = [i for i, name in enumerate(feature_names) if name in self.selected_variables]
            X_final = X[:, selected_idx]
            
            try:
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                lr.fit(X_final, y)
                self.global_model_coefficients = lr.coef_
            except Exception as e:
                warnings.warn(f"Error fitting global model: {e}")
                self.global_model_coefficients = None
        
        print("--- Interface Global Update Complete ---\n")
        
        return {
            'selected_variables': self.selected_variables,
            'hyperparameters': self.best_hyperparameters,
            'variable_counts': var_counts,
            'n_adapter_bars': len(significant_vars_per_bar)
        }

    def local_adaptation(self, adapter_data: pd.DataFrame) -> Dict:
        """
        Perform local adaptation and structural break detection.
        
        This implements:
        1. Fitting adaLASSO on current adapter time bar
        2. Nested testing for new variables not in global selection
        3. Structural break detection based on rejection count
        
        Args:
            adapter_data: DataFrame for current adapter time bar
            
        Returns:
            Dictionary with adaptation results and structural break flag
        """
        if self.selected_variables is None:
            raise ValueError("No global model available. Run global_update() first.")
        
        print("--- Starting Interface Local Adaptation ---")
        
        try:
            # Prepare data for current adapter bar
            X, y, feature_names = self._prepare_data(adapter_data)
            
            if len(X) < 5:  # Need minimum observations
                return {
                    'structural_break': False,
                    'reason': 'insufficient_data',
                    'n_observations': len(X)
                }
            
            # Fit adaLASSO with global hyperparameters
            enet_result = self._fit_enet(X, y, alpha=self.best_hyperparameters['alpha'])
            ada_result = self._fit_ada_lasso(
                X, y, enet_result['coefficients'],
                lambda_ada=self.best_hyperparameters['lambda_ada'],
                gamma=self.best_hyperparameters['gamma']
            )
            
            # Get variables selected by local adaLASSO
            local_selected = np.array(feature_names)[ada_result['selected_variables']]
            
            # Find new variables not in global selection
            new_variables = [var for var in local_selected if var not in self.selected_variables]
            
            if not new_variables:
                # No new variables, no structural break
                print(f"No new variables selected. Current selections: {len(local_selected)}")
                return {
                    'structural_break': False,
                    'reason': 'no_new_variables',
                    'local_selected': local_selected.tolist(),
                    'new_variables': []
                }
            
            # Nested test for new variables
            print(f"Testing {len(new_variables)} new variables: {new_variables}")
            
            # Test if adding new variables significantly improves model
            # Restricted model: only global variables
            global_var_idx = [i for i, name in enumerate(feature_names) if name in self.selected_variables]
            new_var_idx = [i for i, name in enumerate(feature_names) if name in new_variables]
            
            if not global_var_idx:
                # No global variables available for comparison
                print("No global variables available for nested test")
                self.adapter_rejections += 1
            else:
                try:
                    from scipy.stats import f
                    import statsmodels.api as sm
                    
                    # Restricted model (global variables only)
                    X_restricted = sm.add_constant(X[:, global_var_idx])
                    model_restricted = sm.OLS(y, X_restricted).fit()
                    
                    # Unrestricted model (global + new variables)
                    X_unrestricted = sm.add_constant(X[:, global_var_idx + new_var_idx])
                    model_unrestricted = sm.OLS(y, X_unrestricted).fit()
                    
                    # F-test for nested models
                    rss_restricted = model_restricted.ssr
                    rss_unrestricted = model_unrestricted.ssr
                    
                    df_restricted = len(y) - len(global_var_idx) - 1
                    df_unrestricted = len(y) - len(global_var_idx) - len(new_var_idx) - 1
                    
                    if df_unrestricted > 0 and rss_unrestricted > 0:
                        f_stat = ((rss_restricted - rss_unrestricted) / len(new_var_idx)) / (rss_unrestricted / df_unrestricted)
                        p_value = 1 - f.cdf(f_stat, len(new_var_idx), df_unrestricted)
                        
                        # Reject null if new variables are significant
                        if p_value < self.significance_threshold:
                            self.adapter_rejections += 1
                            print(f"Nested test rejected (p={p_value:.4f}). Rejection count: {self.adapter_rejections}")
                        else:
                            print(f"Nested test not rejected (p={p_value:.4f})")
                            print(f"Nested test not rejected (p={p_value:.4f})")
                    else:
                        # Degenerate case
                        print("Degenerate nested test case")
                        
                except Exception as e:
                    warnings.warn(f"Error in nested test: {e}")
                    # Conservative approach: count as rejection
                    self.adapter_rejections += 1
            
            # Check for structural break
            structural_break = self.adapter_rejections >= self.structural_break_threshold
            
            if structural_break:
                print(f"STRUCTURAL BREAK DETECTED! Rejections: {self.adapter_rejections} >= {self.structural_break_threshold}")
            
            print("--- Interface Local Adaptation Complete ---\n")
            
            return {
                'structural_break': structural_break,
                'reason': 'rejection_threshold_exceeded' if structural_break else 'normal_adaptation',
                'local_selected': local_selected.tolist(),
                'new_variables': new_variables,
                'rejection_count': self.adapter_rejections,
                'threshold': self.structural_break_threshold
            }
            
        except Exception as e:
            warnings.warn(f"Error in local adaptation: {e}")
            return {
                'structural_break': False,
                'reason': 'error',
                'error': str(e)
            }

    def reset_adaptation_state(self):
        """Reset the adaptation state (rejection count) after structural break handling."""
        print("Resetting interface adaptation state...")
        self.adapter_rejections = 0

    def get_selected_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get data with only the selected variables.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with only selected variables plus timestamp and target
        """
        if self.selected_variables is None:
            raise ValueError("No variables selected. Run global_update() first.")
        
        # Keep essential columns
        essential_cols = ['timestamp', 'micro_return']
        keep_cols = essential_cols + self.selected_variables
        
        # Filter to existing columns
        available_cols = [col for col in keep_cols if col in data.columns]
        
        return data[available_cols].copy()

    

