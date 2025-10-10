import copy
import logging
import os
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Activate pandas to R conversion
pandas2ri.activate()

rugarch = importr('rugarch')

class Adapter:
    def __init__(self, 
                 interface,
                 update_method: str = 'myopic',
                 arma_order : Tuple[int, int] = (1, 1),
                 garch_order: Tuple[int, int] = (1, 1),
                 dist: str = 'norm', # If called should fail 
                 model : str = 'sGARCH',
                 archm: bool = False,
                 parameters : dict = {},
                 compare_with_cnn_lstm: bool = False):
        self.interface = interface
        self.update_method = update_method
        self.arma_order = arma_order
        self.garch_order = garch_order
        self.dist = dist
        self.model = model
        self.archm = archm
        self.parameters = parameters
        self.compare_with_cnn_lstm = compare_with_cnn_lstm
        self.spec = None  # To store GARCH specification
        self.logger = logging.getLogger('ttstl.adapter')

    def _update_myopic(self, new_parameters: dict):
        self.parameters = new_parameters

    # Combining update rules
    def _update_simple_average(self, new_parameters: dict, ell : int):
        if not self.parameters:
            self.parameters = new_parameters
        else:
            # Simple average of current and new parameters
            for key in new_parameters:
                if key in self.parameters:
                    self.parameters[key] = (ell - 1) / ell * self.parameters[key] + \
                        1 / ell * new_parameters[key]
    
    def _update_power_law(self, new_parameters : dict, ell : int, alpha : float = 3):
        if not self.parameters:
            self.parameters = new_parameters
        else:
            for key in new_parameters:
                ell_seq = np.arange(1, ell) ** alpha
                sum_ell_seq = np.sum(ell_seq)
                if key in self.parameters:
                    self.parameters[key] = sum_ell_seq / (sum_ell_seq + ell ** alpha) * self.parameters[key] + \
                                           ell ** alpha / (sum_ell_seq + ell ** alpha) * new_parameters[key]

    def _update_exponential_smoothing(self, new_parameters : dict, alpha : float = 0.1):
        if not self.parameters:
            self.parameters = new_parameters
        else:
            for key in new_parameters:
                if key in self.parameters:
                    self.parameters[key] = (1 - alpha) * self.parameters[key] + alpha * new_parameters[key]

    def _update_minimum_variance(self):
        pass

    # Recursive update rules
    def _update_newton_raphson(self, new_parameters: dict, 
                               gradient: np.ndarray = None, 
                                hessian: np.ndarray = None, 
                                regularization: float = 0.0):
        """
        Newton-Raphson parameter update: θ_new = θ_old - (H + λ I)^(-1) * ∇f
        
        Parameters:
        -----------
        new_parameters : dict - Current fitted parameters (θ_old)
        gradient : np.ndarray - Gradient vector from numderiv_garch
        hessian : np.ndarray - Hessian matrix from numderiv_garch  
        param_names : list - Parameter names ensuring consistent ordering
        regularization : float - Diagonal regularization for numerical stability (λ)
        """
        
        if not self.parameters:
            self.parameters = new_parameters
        else:
            self.logger.info(gradient)
            self.logger.info(hessian)
            # 1. Validate inputs and convert to arrays
            current_params_array = np.array([self.parameters[name] for name in list(new_parameters.keys())])

            # 2. Regularize hessian for numerical stability
            regularized_hessian = hessian + regularization * np.eye(len(hessian))
            
            # 3. Compute Newton step using stable linear solver
            try:
                newton_step = np.linalg.solve(regularized_hessian, gradient)
            except np.linalg.LinAlgError as e:
                self.logger.error(f"Failed to compute Newton step due to singular matrix: {e}. Falling back to zero step.")
                # Fallback to a zero step to prevent update failure
                newton_step = np.zeros_like(gradient)
                
            # 4. Apply update with step size control
            updated_params_array = current_params_array - newton_step
            
            # 5. Convert back to dictionary maintaining order
            self.parameters = dict(zip(list(new_parameters.keys()), updated_params_array))


    def _update_fischer_scoring(self, new_parameters: dict, 
                                gradient: np.ndarray = None, 
                                info_mat: np.ndarray = None, 
                                regularization: float = 0.0):
        """
        Fisher scoring parameter update: θ_new = θ_old + (I_FS + λ I)^(-1) * ∇f

        Parameters:
        -----------
        new_parameters : dict - Current fitted parameters (θ_old)
        gradient : np.ndarray - Gradient vector from numderiv_garch
        hessian : np.ndarray - Hessian matrix from numderiv_garch  
        param_names : list - Parameter names ensuring consistent ordering
        regularization : float - Diagonal regularization for numerical stability (λ)
        """

        if not self.parameters:
            self.parameters = new_parameters
        else:
            # 1. Validate inputs and convert to arrays
            current_params_array = np.array([self.parameters[name] for name in new_parameters.keys()])

            # 2. Regularize information matrix for numerical stability
            regularized_info_mat = info_mat + regularization * np.eye(len(info_mat))

            # 3. Compute Fisher scoring step using stable linear solver
            try:
                fs_step = np.linalg.solve(regularized_info_mat, gradient)
            except np.linalg.LinAlgError as e:
                self.logger.error(f"Failed to compute Fisher scoring step due to singular matrix: {e}. Falling back to zero step.")
                # Fallback to a zero step to prevent update failure
                fs_step = np.zeros_like(gradient)

            # 4. Apply update with step size control
            updated_params_array = current_params_array + fs_step

            # 5. Convert back to dictionary maintaining order
            self.parameters = dict(zip(list(new_parameters.keys()), updated_params_array))

    def _update_gradient_ascent(self, 
                                new_parameters: dict, 
                                gradient: np.ndarray = None, 
                                alpha: float = 0.01):
        """
        Gradient ascent parameter update: θ_new = θ_old + α ∇f

        Parameters:
        -----------
        new_parameters : dict - Current fitted parameters (θ_old)
        gradient : np.ndarray - Gradient vector from numderiv_garch
        param_names : list - Parameter names ensuring consistent ordering
        alpha : float - Step size for gradient descent
        """

        if not self.parameters:
            self.parameters = new_parameters
        else:
            # 1. Validate inputs and convert to arrays
            current_params_array = np.array([self.parameters[name] for name in new_parameters.keys()])

            # 2. Compute gradient ascent step
            ga_step = alpha * gradient

            # 3. Apply update with step size control
            updated_params_array = current_params_array + ga_step

            # 4. Convert back to dictionary maintaining order
            self.parameters = dict(zip(list(new_parameters.keys()), updated_params_array))

    def _print_parameters_summary(self, adapter_bar_number: int, new_parameters: dict, garch_results: dict):
        """
        Print a Python summary of the GARCH model parameters.
        
        Parameters:
        -----------
        adapter_bar_number : int
            The adapter bar number for labeling
        new_parameters : dict
            Dictionary containing the newly fitted parameters
        garch_results : dict
            Dictionary containing GARCH fitting results including standard errors and p-values
        """
        self.logger.info(f"=" * 80)
        self.logger.info(f"GARCH Model Parameters")
        self.logger.info(f"=" * 80)
        self.logger.info(f"Update method: {self.update_method}")

        # Get coefficient names, standard errors, and p-values from garch_results
        coef_names = garch_results['coefficient_names']
        coef_values = garch_results['coefficients']
        se_values = garch_results['se_coefficients']
        p_values = garch_results['p_values']
        
        # Create mapping from mxreg to actual variable names
        selected_vars = self.interface.selected_variables
        mxreg_mapping = {}
        mxreg_count = 1
        for name in coef_names:
            if name.startswith('mxreg'):
                if mxreg_count <= len(selected_vars):
                    mxreg_mapping[name] = selected_vars[mxreg_count - 1]
                    mxreg_count += 1
                else:
                    mxreg_mapping[name] = name  # fallback if mapping fails
            else:
                mxreg_mapping[name] = name
        
        # Print adapter bar-specific estimates (current fit)
        self.logger.info(f"Bar Specific Estimates")
        self.logger.info(f"{'Parameter':<20} {'Coefficient':>12} {'Std Error':>12} {'P-Value':>10}")
        self.logger.info("=" * 80)

        for i, (name, coef, se, pval) in enumerate(zip(coef_names, coef_values, se_values, p_values)):
            display_name = mxreg_mapping[name]
            self.logger.info(f"{display_name:<20} {coef:>12.6f} {se:>12.6f} {pval:>10.4f}")
        
        self.logger.info("=" * 80)

        # Print carried estimates (averaged parameters)
        self.logger.info(f"Carried Estimates:")
        self.logger.info(f"{'Parameter':<20} {'Coefficient':>12}")
        self.logger.info("=" * 80)

        for name in coef_names:
            display_name = mxreg_mapping[name]
            carried_value = self.parameters.get(name, 0.0)
            self.logger.info(f"{display_name:<20} {carried_value:>12.6f}")

        self.logger.info("=" * 80)

    def _fit_garch_model(self, y : np.ndarray, 
                         X : np.ndarray = None, 
                         Z : np.ndarray = None, out_sample=0):
        """
        Fit GARCH model using R's rugarch package through rpy2.
        
        Parameters:
        -----------
        y : array-like
            The time series data to fit
        X : array-like, optional
            External regressors for the mean model
        Z : array-like, optional  
            External regressors for the variance model
        out_sample : int, optional
            Number of out-of-sample observations
            
        Returns:
        --------
        dict : Dictionary containing GARCH model results
        """
        #try:
        # Load the R script with run_garch function
        r_script_path = os.path.join(os.path.dirname(__file__), 'r_scripts', 'run_garch.R')
        ro.r(f'source("{r_script_path}")')
        run_garch_model = ro.globalenv['run_garch']
        
        # Convert Python data to R
        r_y = ro.FloatVector(y)
        
        # Handle optional external regressors
        r_X = ro.NULL if X is None else pandas2ri.py2rpy(pd.DataFrame(X))
        r_Z = ro.NULL if Z is None else pandas2ri.py2rpy(pd.DataFrame(Z))
        
        # Create parameter dictionary for R function
        r_params = {
            'y': y,
            'X': X,
            'Z': Z,
            'model': self.model,
            'p': self.arma_order[0],
            'q': self.arma_order[1], 
            'm': self.garch_order[0],
            's': self.garch_order[1],
            'archm': self.archm,
            'dist': self.dist,
            'out_sample': out_sample,
            'display': False  # Don't generate plots in batch processing
        }
        
        if self.parameters:
           r_params['start_pars'] = ro.ListVector(self.parameters)


        # Call the R function
        r_result = run_garch_model(y = y,
                        X = ro.NULL if X is None else X,
                        Z = ro.NULL if Z is None else Z,
                        p = self.arma_order[0],
                        q = self.arma_order[1],
                        m = self.garch_order[0],
                        s = self.garch_order[1],
                        archm = self.archm,
                        dist = self.dist,
                        model = self.model,
                        start_pars = ro.ListVector({}) if not self.parameters else r_params['start_pars'],
                        out_sample = out_sample,
                        display = False)

        # Convert R results back to Python dictionary
        result_dict = {}
        
        # Extract key results from R list
        result_dict['coefficients'] = np.array(r_result.rx2('coef'))
        result_dict['se_coefficients'] = np.array(r_result.rx2('se_coef'))
        result_dict['p_values'] = np.array(r_result.rx2('pvals'))
        result_dict['coefficient_names'] = list(r_result.rx2('coef_names'))
        result_dict['gof_pval'] = np.array(r_result.rx2('gof'))
        result_dict['lb_pval'] = np.array(r_result.rx2('lb'))
        result_dict['lb_sq_pval'] = np.array(r_result.rx2('lb_sq'))
        result_dict['standardized_residuals'] = np.array(r_result.rx2('stdresid'))
        result_dict['saddle_point_indicator'] = int(r_result.rx2('saddle')[0])
        result_dict['spec'] = r_result.rx2('spec')  # Store GARCH specification

        # Store model specification info
        result_dict['model_info'] = {
            'model': self.model,
            'arma_order': self.arma_order,
            'garch_order': self.garch_order,
            'distribution': self.dist,
            'archm': self.archm
        }
        
        return result_dict
            
        # except Exception as e:
        #     raise RuntimeError(f"Failed to fit GARCH model: {str(e)}")

    def get_nn_output(self,
                      event_driven_adapter_time_bar_data: pd.DataFrame,
                      physical_time_driven_adapter_time_bar_data: pd.DataFrame,
                      batch_size: int = 512) -> pd.DataFrame:
        """
        Extract raw neural network outputs (forward pass) from the backbone model.
        
        This method uses the backbone's extract_neural_covariates method with get_output=True
        to obtain the raw CNN-LSTM outputs instead of hidden activations. This is useful
        for comparison purposes and analysis of the model's direct predictions.
        
        Args:
            event_driven_adapter_time_bar_data (pd.DataFrame): Event-driven data for neural network processing.
            physical_time_driven_adapter_time_bar_data (pd.DataFrame): Physical time-driven data with timestamps.
            batch_size (int, optional): Batch size for processing. Defaults to 512.
            
        Returns:
            pd.DataFrame: DataFrame containing timestamps and raw neural network outputs with 'forward_pass_' prefix.
            
        Example:
            >>> nn_outputs = adapter.get_nn_output(event_df, phys_df)
            >>> print(nn_outputs.columns)
            ['timestamp', 'forward_pass_0']  # For single output CNN-LSTM
        """
        self.logger.info("Extracting raw neural network outputs from backbone")
        
        # Use the backbone's extract_neural_covariates method with get_output=True
        # This will return the forward pass outputs instead of hidden activations
        nn_outputs = self.interface.backbone.extract_neural_covariates(
            event_driven_data=event_driven_adapter_time_bar_data,
            physical_time_data=physical_time_driven_adapter_time_bar_data,
            batch_size=batch_size,
            get_output=True  # This is the key parameter to get raw outputs
        )
        
        # Convert timestamp to float for consistency
        nn_outputs['timestamp'] = nn_outputs['timestamp'].astype(float)
        
        # Log information about the extracted outputs
        output_columns = [col for col in nn_outputs.columns if col.startswith('forward_pass_')]
        self.logger.info(f"Extracted {len(output_columns)} neural network output features: {output_columns}")
        self.logger.info(f"Neural network outputs shape: {nn_outputs.shape}")
        
        return nn_outputs

    def augment_data(self,
                     event_driven_adapter_time_bar_data : pd.DataFrame,
                     physical_time_driven_adapter_time_bar_data : pd.DataFrame,
                     scaler : Optional[StandardScaler] = None,
                     target_col_name : str = 'micro_return'):
        """Augments the physical_time_driven_adapter_time_bar_data with neural covariates."""

        self.logger.info("Augmenting physical time data with neural covariates")

        neural_covariates = self.interface.backbone.extract_neural_covariates(event_driven_adapter_time_bar_data,
                                                                              physical_time_driven_adapter_time_bar_data)
        
        # Ensure timestamp consistency - convert to float and create copy to avoid modifying original
        neural_covariates['timestamp'] = neural_covariates['timestamp'].astype(float)
        physical_data_copy = physical_time_driven_adapter_time_bar_data.copy()
        physical_data_copy['timestamp'] = physical_data_copy['timestamp'].astype(float)

        # Merge neural covariates with physical time data
        augmented_physical_data = pd.merge_asof(
            physical_data_copy,
            neural_covariates,
            on='timestamp',
            direction='nearest',
            tolerance=1e-6
        )

        covariate_names = [var for var in augmented_physical_data.columns if var not in ['timestamp', 
                                                                                         'adapter_bar', 
                                                                                         'backbone_bar', 
                                                                                         target_col_name]]

        
        if scaler is not None:
            self.logger.info(f"Scaling {len(covariate_names)} covariates")
            
            # Smart column reordering and validation for scaler compatibility
            if hasattr(scaler, 'feature_names_in_'):
                fitted_columns = scaler.feature_names_in_.tolist()
                
                # Check for missing columns (critical error)
                missing_cols = set(fitted_columns) - set(covariate_names)
                if missing_cols:
                    self.logger.error(f"Cannot scale: missing required columns: {missing_cols}")
                    self.logger.error(f"Available columns: {covariate_names}")
                    self.logger.error(f"Required columns: {fitted_columns}")
                    raise ValueError(f"Missing required columns for scaling: {missing_cols}")
                
                # Check for extra columns (will be ignored)
                extra_cols = set(covariate_names) - set(fitted_columns)
                if extra_cols:
                    self.logger.warning(f"Ignoring extra columns not in fitted scaler: {extra_cols}")
                
                # Automatically reorder columns to match scaler's fitted order
                original_order = covariate_names.copy()
                reordered_columns = fitted_columns  # Use exact fitted order
                
                self.logger.info(f"Automatically reordered {len(reordered_columns)} columns to match scaler:")
                self.logger.info(f"  Original order: {original_order}")
                self.logger.info(f"  Reordered to:   {reordered_columns}")
                
                # Apply scaling with properly ordered columns
                augmented_physical_data[reordered_columns] = scaler.transform(augmented_physical_data[reordered_columns])
                
            else:
                # Fallback for older scikit-learn versions without feature_names_in_
                self.logger.warning("Scaler does not have feature_names_in_ attribute.")
                self.logger.warning("Using original column order - this may cause scaling errors!")
                augmented_physical_data[covariate_names] = scaler.transform(augmented_physical_data[covariate_names])

        self.logger.info(f"Merged neural covariates")

        return augmented_physical_data

    def update_adapter(self, 
                       event_driven_adapter_time_bar_data: pd.DataFrame,
                       physical_time_driven_adapter_time_bar_data: pd.DataFrame,
                       scaler : StandardScaler,
                       adapter_bar_number: int,
                       target_col_name: str = 'micro_return',
                       update_parameters: Optional[dict] = None) -> dict:
        """
        Update adapter parameters by fitting ARMAX-GARCH model on new data from an adapter time bar.
        
        Parameters:
        -----------
        event_driven_adapter_time_bar_data : pd.DataFrame
            Event-driven data for neural covariate extraction
        physical_time_driven_adapter_time_bar_data : pd.DataFrame
            Physical time-driven data for a specific adapter time bar containing the target variable
        adapter_bar_number : int
            The adapter bar number (used for averaging schemes)
        target_col_name : str, optional
            Name of the target variable column (default: 'micro_return')
        update_parameters : dict, optional
            Additional parameters for the update method
            
        Returns:
        --------
        dict : Dictionary containing GARCH fitting results and updated parameters
        """
        if update_parameters is None:
            update_parameters = {}
            
        # Validate that interface has selected variables (required for proper operation)
        if not hasattr(self.interface, 'selected_variables') or self.interface.selected_variables is None:
            raise ValueError("Interface must have selected variables before adapter update. "
                           "Run interface.global_update() first.")
        
        # Validate target column exists in physical data
        if target_col_name not in physical_time_driven_adapter_time_bar_data.columns:
            raise ValueError(f"Target column '{target_col_name}' not found in physical time data")
        
        self.logger.info('='*80)
        self.logger.info('='*80)
        self.logger.info("Running adapter module update")
        self.logger.info('='*80)
        self.logger.info('='*80)
        
        # Use interface's _prepare_data method to augment data with neural covariates and lagged variables
        self.logger.info(f"Preparing data for adapter bar {adapter_bar_number}")
        try:
            augmented_data = self.augment_data(
                event_driven_adapter_time_bar_data=event_driven_adapter_time_bar_data,
                physical_time_driven_adapter_time_bar_data=physical_time_driven_adapter_time_bar_data,
                scaler=scaler,
                target_col_name=target_col_name
            )
            if augmented_data.isnull().any(axis = 1).sum() == 1:
                self.logger.info(f"Dropping the first sampling time bar due to NaNs")
                augmented_data = augmented_data.iloc[1:]
            elif augmented_data.isnull().any(axis=1).sum() > 1:
                augmented_data = augmented_data.iloc[1:]
                self.logger.warning(f"Remaining data contains {augmented_data.isnull().any(axis=1).sum()} NaNs.")
            else:
                self.logger.info(f"All remaining data contains no NaNs")
            
        except Exception as e:
            self.logger.error(f"Failed to prepare data for adapter bar {adapter_bar_number}: {str(e)}")
            raise RuntimeError(f"Failed to prepare data for adapter bar {adapter_bar_number}: {str(e)}")
        
        # Filter data to only the specific adapter bar if needed
        if 'adapter_bar' in augmented_data.columns:
            # Find the adapter bar that corresponds to our data
            unique_bars = augmented_data['adapter_bar'].unique()
            if len(unique_bars) > 1:
                # Use the last adapter bar if multiple exist
                self.logger.warning(f"Multiple adapter bars found in data: {unique_bars}. Using the last one.")
                target_adapter_bar = unique_bars[-1]
                adapter_bar_data = augmented_data[augmented_data['adapter_bar'] == target_adapter_bar].copy()
                self.logger.info(f"Filtered to adapter bar {target_adapter_bar}, {len(adapter_bar_data)} observations")
            else:
                adapter_bar_data = augmented_data.copy()
                self.logger.info(f"Using all data: {len(adapter_bar_data)} observations")
        else:
            adapter_bar_data = augmented_data.copy()
            self.logger.info(f"No adapter_bar column found, using all data: {len(adapter_bar_data)} observations")

        # Extract target variable
        if target_col_name not in adapter_bar_data.columns:
            self.logger.error(f"Target column '{target_col_name}' not found in augmented data")
            raise ValueError(f"Target column '{target_col_name}' not found in augmented data")
        
        y = adapter_bar_data[target_col_name]
        
        # Get selected variables from interface
        selected_vars = self.interface.selected_variables
        lags_prefix = target_col_name + '_lag_'
        # Filter out lag variables from selected variables
        selected_vars = [var for var in selected_vars if not var.startswith(lags_prefix)]
        self.logger.info(f"Selected variables from interface: {selected_vars}")

        # Check which selected variables are available in the augmented data
        missing_vars = [var for var in selected_vars if var not in adapter_bar_data.columns]
        
        if missing_vars:
            self.logger.error(f"Selected variables not found in augmented data: {missing_vars}")
            raise ValueError(f"Selected variables not found in augmented data: {missing_vars}")
        
        # Prepare external regressors
        X = adapter_bar_data[selected_vars]
        
        if X.isnull().any().any():
            self.logger.error("External regressors contain NaN values. Please ensure all selected variables are present and clean.")
            raise ValueError("External regressors contain NaN values. Please ensure all selected variables are present and clean.")

        if len(y) < 10:
            self.logger.error(f"After removing NaN values, insufficient data: {len(y)} observations")
            raise ValueError(f"After removing NaN values, insufficient data: {len(y)} observations")
        
        # Fit GARCH model
        try:
            self.logger.info(f"Fitting GARCH model with {len(y)} observations and {len(selected_vars)} covariates...")
            #self.logger.info(f"Target variable (y) head:\n{y.head()}")
            #self.logger.info(f"External regressors (X) head:\n{X.head()}")
            garch_results = self._fit_garch_model(
                y=y.values,
                X=X.values,
                Z=None,  # No variance regressors for now
                out_sample=0
            )
            
            # Extract fitted parameters
            coef_names = copy.deepcopy(garch_results['coefficient_names'])
            coef_values = copy.deepcopy(garch_results['coefficients'])
            new_parameters = dict(zip(coef_names, coef_values))

            self.logger.info(f"GARCH model fitted successfully")

            self.spec = garch_results['spec']  # Store GARCH specification

            if self.parameters:
                # Check if the new parameters have the same keys as the existing ones
                if set(new_parameters.keys()) != set(self.parameters.keys()):
                    self.logger.warning("Parameter mismatch between new and existing parameters.")
                    self.logger.warning(f"Existing keys: {sorted(list(self.parameters.keys()))}")
                    self.logger.warning(f"New keys: {sorted(list(new_parameters.keys()))}")
                    # This might indicate a change in model structure, which could be problematic
                    # for some update rules. For now, we'll log a warning.

                # Load the R script with numderiv_garch function
                r_script_path = os.path.join(os.path.dirname(__file__), 'r_scripts', 'numderiv_garch.R')
                ro.r(f'source("{r_script_path}")')
                numderiv_garch = ro.globalenv['numderiv_garch']
                theta = copy.deepcopy(self.parameters)
                # Call the numderiv_garch function
                numderiv_results = numderiv_garch(
                    spec=self.spec,
                    y=y.values,
                    theta=ro.ListVector(theta)
                )

                grad = np.array(numderiv_results.rx2('grad'))
                hessian = np.array(numderiv_results.rx2('hessian'))
                info_mat = np.array(numderiv_results.rx2('I'))


            else:
                grad = None
                hessian = None
                info_mat = None


            # Update parameters based on update method
            if self.update_method == 'myopic':
                self._update_myopic(new_parameters)
            elif self.update_method == 'simple_average':
                self._update_simple_average(new_parameters, adapter_bar_number)
            elif self.update_method == 'power_law':
                alpha = update_parameters['alpha']
                self._update_power_law(new_parameters, adapter_bar_number, alpha=alpha)
            elif self.update_method == 'exponential_smoothing':
                alpha = update_parameters['alpha']
                self._update_exponential_smoothing(new_parameters, alpha=alpha)
            elif self.update_method == 'newton_raphson':
                regularization = 0.0
                if update_parameters:
                    regularization = update_parameters['regularization'] \
                        if 'regularization' in update_parameters else 0.0
                self._update_newton_raphson(new_parameters,
                                            gradient=grad,
                                            hessian=hessian,
                                            regularization=regularization)
            elif self.update_method == 'fischer_scoring':
                regularization = 0.0
                if update_parameters:
                    regularization = update_parameters['regularization'] \
                        if 'regularization' in update_parameters else 0.0
                self._update_fischer_scoring(new_parameters,
                                              gradient=grad,
                                              info_mat=info_mat,
                                              regularization=regularization)
            elif self.update_method == 'gradient_ascent':
                self._update_gradient_ascent(new_parameters,
                                              gradient=grad,
                                              alpha=update_parameters['alpha'])
            else:
                self.logger.info(f"Update method '{self.update_method}' not implemented")
                raise ValueError(f"Update method '{self.update_method}' not implemented")
            
            
            # Print updated parameters summary
            self._print_parameters_summary(adapter_bar_number, new_parameters, garch_results)

            self.logger.info('='*80)
            self.logger.info('='*80)
            self.logger.info("Finished running adapter module update")
            self.logger.info('='*80)
            self.logger.info('='*80)
            
            # Prepare return dictionary
            return garch_results

        except Exception as e:
            raise RuntimeError(f"Failed to update adapter for bar {adapter_bar_number}: {str(e)}")

    def forecast(self,
                 in_sample_event_df: pd.DataFrame,
                 in_sample_phys_df: pd.DataFrame,
                 out_of_sample_event_df: pd.DataFrame,
                 out_of_sample_phys_df: pd.DataFrame,
                 scaler : StandardScaler,
                 target_col_name: str = 'micro_return'):
        """
        Generates a forecast using a previously fitted GARCH model specification.

        This method prepares the combined in-sample and out-of-sample data by
        augmenting it with neural covariates, then uses the stored GARCH
        specification and parameters to perform a rolling forecast on the
        out-of-sample data.

        Args:
            in_sample_event_df (pd.DataFrame): In-sample event-driven data.
            in_sample_phys_df (pd.DataFrame): In-sample physical time data.
            out_of_sample_event_df (pd.DataFrame): Out-of-sample event-driven data.
            out_of_sample_phys_df (pd.DataFrame): Out-of-sample physical time data.
            target_col_name (str): The name of the target variable column.

        Returns:
            pd.DataFrame: A DataFrame containing the mean and volatility forecasts.
        """
        self.logger.info("Starting forecast generation with full data preparation.")

        # Validate prerequisites
        if self.spec is None:
            self.logger.error("GARCH specification (self.spec) is not available. Fit a model first.")
            raise ValueError("GARCH specification (self.spec) is not available. Fit a model first.")
        if not hasattr(self.interface, 'selected_variables'):
            self.logger.error("Interface 'selected_variables' not available. Run global_update first.")
            raise ValueError("Interface 'selected_variables' not available. Run global_update first.")

        n_in_sample = in_sample_phys_df.shape[0]

        combined_events_df = pd.concat([in_sample_event_df, out_of_sample_event_df])
        combined_phys_df = pd.concat([in_sample_phys_df, out_of_sample_phys_df])

        self.logger.info("Preparing combined data for forecasting")
        try:
            # Use the adapter's augment_data method (consistent with update_adapter)
            self.logger.info(f"Augmenting combined adapter bar data for forecasting")
            augmented_df = self.augment_data(
                event_driven_adapter_time_bar_data=combined_events_df,
                physical_time_driven_adapter_time_bar_data=combined_phys_df,
                scaler=scaler,
                target_col_name=target_col_name
            )

            # Critical. At the start of backbone bars, a NaN might arise in the first row. We drop it for simplicity
            if augmented_df.isnull().any(axis = 1).sum() == 1:
                augmented_df = augmented_df.iloc[1:]
                n_in_sample -= 1

            elif augmented_df.isnull().any(axis=1).sum() > 1:
                self.logger.error("More than one NaN found in augmented data. Please check data integrity.")
                raise ValueError("More than one NaN found in augmented data. Please check data integrity.")

            #self.logger.info(f"Augmented combined data shape: {augmented_df.shape}")

        except Exception as e:
            raise RuntimeError(f"Failed to prepare data for forecasting: {e}")

        in_sample_augmented = augmented_df[:n_in_sample]
        out_of_sample_augmented = augmented_df[n_in_sample:]

        # Get selected variables from interface
        selected_vars = self.interface.selected_variables
        lags_prefix = target_col_name + '_lag_'
        # Filter out lag variables from selected variables
        selected_vars = [var for var in selected_vars if not var.startswith(lags_prefix)]

        self.logger.info(f"Selected variables for forecasting: {selected_vars}")
        
        # In-sample data
        in_sample_y = in_sample_augmented[target_col_name]
        #in_sample_X = in_sample_augmented[selected_vars]
        
        # Out-of-sample data
        out_of_sample_y = out_of_sample_augmented[target_col_name]
        out_of_sample_X = out_of_sample_augmented[selected_vars]

        # Combine again for the R function call format
        combined_y = pd.concat([in_sample_y, out_of_sample_y])
        #combined_X = pd.concat([in_sample_X, out_of_sample_X])

        # Perform forecast using R
        self.logger.info(f"Performing rolling forecast on {len(out_of_sample_y)} observations.")
        r_script_path = os.path.join(os.path.dirname(__file__), 'r_scripts', 'run_garch.R')
        ro.r(f'source("{r_script_path}")')
        set_fixed_parameters = ro.globalenv['set_fixed_parameters']

        # Set the GARCH spec with the fixed parameters from the last update
        spec_with_fixed_pars = set_fixed_parameters(self.spec, ro.ListVector(self.parameters))

        # Prepare external regressors for the forecast period
        external_forecasts_r = ro.ListVector({
            'mregfor': ro.r['matrix'](
                out_of_sample_X.values,
                nrow=out_of_sample_X.shape[0],
                ncol=out_of_sample_X.shape[1]
            )
        })

        # Call the ugarchforecast function
        garch_forecast = rugarch.ugarchforecast(
            fitORspec=spec_with_fixed_pars,
            data=combined_y.values,
            n_ahead=1,
            n_roll=len(out_of_sample_y) - 1,
            out_sample=len(out_of_sample_y),
            external_forecasts=external_forecasts_r
        )

        # Process and return results
        fore_mean = rugarch.fitted(garch_forecast)
        fore_vol = rugarch.sigma(garch_forecast)

        forecast_df = pd.DataFrame({
            'timestamp': out_of_sample_augmented['timestamp'],
            'mean': np.array(fore_mean).flatten(),
            'vol': np.array(fore_vol).flatten()
        })
        
        self.logger.info("Forecast generation complete.")
        return forecast_df
    
    def calculate_performance_metrics(self,
                          in_sample_event_df: pd.DataFrame,
                          in_sample_phys_df: pd.DataFrame,
                          out_of_sample_event_df: pd.DataFrame,
                          out_of_sample_phys_df: pd.DataFrame,
                          benchmark_forecast: Union[float, pd.Series],
                          scaler : StandardScaler,
                          target_col_name: str = 'micro_return'):
        """
        Calculates Mean Absolute Error (MAE) and Mean Squared Error (MSE)
        by leveraging the new forecast method and optionally comparing with raw neural network outputs.

        Args:
            in_sample_event_df (pd.DataFrame): In-sample event-driven data.
            in_sample_phys_df (pd.DataFrame): In-sample physical time data.
            out_of_sample_event_df (pd.DataFrame): Out-of-sample event-driven data.
            out_of_sample_phys_df (pd.DataFrame): Out-of-sample physical time data.
            benchmark_forecast (Union[float, pd.Series]): Benchmark forecast values.
            scaler (StandardScaler): Fitted scaler for data preprocessing.
            target_col_name (str): The name of the target variable column.

        Returns:
            dict: A dictionary containing the calculated metrics for GARCH model and optionally for raw NN outputs.
        """
        if self.compare_with_cnn_lstm:
            self.logger.info("Calculating MAE and MSE using the forecast method and raw neural network outputs.")
        else:
            self.logger.info("Calculating MAE and MSE using the forecast method only (CNN-LSTM comparison disabled).")

        # Call the GARCH forecast method to get predictions
        garch_forecast_df = self.forecast(
            in_sample_event_df=in_sample_event_df,
            in_sample_phys_df=in_sample_phys_df,
            out_of_sample_event_df=out_of_sample_event_df,
            out_of_sample_phys_df=out_of_sample_phys_df,
            scaler=scaler,
            target_col_name=target_col_name
        )

        # Conditionally get raw neural network outputs for out-of-sample period
        if self.compare_with_cnn_lstm:
            self.logger.info("Extracting raw neural network outputs for performance comparison")
            nn_outputs = self.get_nn_output(
                event_driven_adapter_time_bar_data=out_of_sample_event_df,
                physical_time_driven_adapter_time_bar_data=out_of_sample_phys_df,
                batch_size=512
            )

            # Handle NaN values in nn_outputs - replace with zeros and report
            nan_count = nn_outputs.isna().sum().sum()
            if nan_count > 0:
                nn_outputs = nn_outputs.fillna(0)
                self.logger.warning(f"Replaced {nan_count} NaN values with zeros in neural network outputs")
            else:
                self.logger.info("No NaN values found in neural network outputs")
        else:
            self.logger.info("Skipping neural network output extraction (compare_with_cnn_lstm=False)")
            nn_outputs = None

        # Ensure timestamp consistency and alignment between GARCH forecasts and actual values
        garch_forecast_df_copy = garch_forecast_df[['timestamp', 'mean']].copy()
        out_of_sample_phys_df_copy = out_of_sample_phys_df[['timestamp', target_col_name]].copy()
        
        # Convert timestamps to float for consistent merging
        garch_forecast_df_copy['timestamp'] = garch_forecast_df_copy['timestamp'].astype(float)
        out_of_sample_phys_df_copy['timestamp'] = out_of_sample_phys_df_copy['timestamp'].astype(float)
        
        garch_forecast_with_actual = pd.merge_asof(
            garch_forecast_df_copy.sort_values('timestamp'),
            out_of_sample_phys_df_copy.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=1e-6  # Tolerance for float timestamp matching
        )
        
        # Conditionally ensure alignment between NN outputs and actual values
        if self.compare_with_cnn_lstm and nn_outputs is not None:
            # nn_outputs already has float timestamps from get_nn_output method
            nn_output_with_actual = pd.merge_asof(
                nn_outputs.sort_values('timestamp'),
                out_of_sample_phys_df_copy.sort_values('timestamp'),
                on='timestamp',
                direction='nearest',
                tolerance=1e-6
            )
        else:
            nn_output_with_actual = None
        
        # Extract aligned data for GARCH model
        garch_actual_values = garch_forecast_with_actual[target_col_name]
        garch_predicted_values = garch_forecast_with_actual['mean']
        
        # Conditionally extract aligned data for NN outputs
        if self.compare_with_cnn_lstm and nn_output_with_actual is not None:
            nn_output_cols = [col for col in nn_output_with_actual.columns if col.startswith('forward_pass_')]
            if len(nn_output_cols) == 0:
                self.logger.warning("No forward_pass columns found in NN outputs. Skipping NN performance metrics.")
                nn_actual_values = None
                nn_predicted_values = None
            else:
                nn_prediction_col = nn_output_cols[0]  # Use first output as prediction
                nn_actual_values = nn_output_with_actual[target_col_name]
                nn_predicted_values = nn_output_with_actual[nn_prediction_col]
                self.logger.info(f"Using NN output column '{nn_prediction_col}' as prediction")
        else:
            nn_actual_values = None
            nn_predicted_values = None
        
        self.logger.info(f"GARCH aligned data: {len(garch_actual_values)} observations (forecast: {len(garch_forecast_df)}, original: {len(out_of_sample_phys_df)})")
        if self.compare_with_cnn_lstm and nn_actual_values is not None:
            self.logger.info(f"NN aligned data: {len(nn_actual_values)} observations")

        # Calculate GARCH model metrics
        garch_mae = mean_absolute_error(garch_actual_values, garch_predicted_values)
        garch_mse = mean_squared_error(garch_actual_values, garch_predicted_values)

        # Conditionally calculate NN output metrics
        if self.compare_with_cnn_lstm and nn_actual_values is not None and len(nn_actual_values) > 0:
            nn_mae = mean_absolute_error(nn_actual_values, nn_predicted_values)
            nn_mse = mean_squared_error(nn_actual_values, nn_predicted_values)
        else:
            nn_mae = None
            nn_mse = None

        # Handle benchmark computation for GARCH model
        try:
            if isinstance(benchmark_forecast, (pd.Series, np.ndarray)):
                # If benchmark is a series, ensure it aligns with actual_values
                if len(benchmark_forecast) != len(garch_actual_values):
                    self.logger.error(f"Benchmark length ({len(benchmark_forecast)}) != actual values length ({len(garch_actual_values)}). Using benchmark mean.")
                    raise ValueError(f"Benchmark length ({len(benchmark_forecast)}) != actual values length ({len(garch_actual_values)}). Using benchmark mean.")
                else:
                    benchmark_value = benchmark_forecast
            else:
                # Scalar benchmark - will broadcast correctly
                benchmark_value = benchmark_forecast
            
            # Calculate relative metrics for GARCH
            benchmark_errors = np.abs(garch_actual_values - benchmark_value)
            benchmark_mse = np.mean((garch_actual_values - benchmark_value) ** 2)
            
            garch_relmae = garch_mae / np.mean(benchmark_errors) if np.mean(benchmark_errors) != 0 else None
            garch_relmse = garch_mse / benchmark_mse if benchmark_mse != 0 else None
            garch_r2 = 1 - garch_relmse if garch_relmse is not None else None
            
            # Conditionally calculate relative metrics for NN outputs
            if self.compare_with_cnn_lstm and nn_actual_values is not None and len(nn_actual_values) > 0:
                # Use same benchmark but align with NN data
                if isinstance(benchmark_forecast, (pd.Series, np.ndarray)):
                    if len(benchmark_forecast) != len(nn_actual_values):
                        self.logger.warning(f"Benchmark length mismatch for NN data. Using scalar benchmark.")
                        nn_benchmark_value = np.mean(benchmark_forecast) if hasattr(benchmark_forecast, '__len__') else benchmark_forecast
                    else:
                        nn_benchmark_value = benchmark_forecast
                else:
                    nn_benchmark_value = benchmark_forecast
                
                nn_benchmark_errors = np.abs(nn_actual_values - nn_benchmark_value)
                nn_benchmark_mse = np.mean((nn_actual_values - nn_benchmark_value) ** 2)
                
                nn_relmae = nn_mae / np.mean(nn_benchmark_errors) if np.mean(nn_benchmark_errors) != 0 else None
                nn_relmse = nn_mse / nn_benchmark_mse if nn_benchmark_mse != 0 else None
                nn_r2 = 1 - nn_relmse if nn_relmse is not None else None
            else:
                nn_relmae = None
                nn_relmse = None
                nn_r2 = None
            
        except Exception as e:
            self.logger.warning(f"Benchmark computation failed: {e}. Setting relative metrics to None.")
            garch_relmae = None
            garch_relmse = None
            garch_r2 = None
            nn_relmae = None
            nn_relmse = None
            nn_r2 = None

        # Log results
        self.logger.info(f"GARCH Model Metrics - MAE: {garch_mae:.6f}, MSE: {garch_mse:.6f}, RelMAE: {garch_relmae:.6f}, RelMSE: {garch_relmse:.6f}, R2: {garch_r2:.6f}")
        if self.compare_with_cnn_lstm and nn_mae is not None:
            self.logger.info(f"NN Output Metrics - MAE: {nn_mae:.6f}, MSE: {nn_mse:.6f}, RelMAE: {nn_relmae:.6f}, RelMSE: {nn_relmse:.6f}, R2: {nn_r2:.6f}")
            
            # Log comparison
            mae_improvement = ((nn_mae - garch_mae) / nn_mae * 100) if nn_mae != 0 else 0
            mse_improvement = ((nn_mse - garch_mse) / nn_mse * 100) if nn_mse != 0 else 0
            self.logger.info(f"GARCH vs NN Comparison - MAE improvement: {mae_improvement:.2f}%, MSE improvement: {mse_improvement:.2f}%")
        elif self.compare_with_cnn_lstm:
            self.logger.info("NN Output Metrics - Not available (no valid NN outputs)")
        else:
            self.logger.info("NN Output Metrics - Skipped (compare_with_cnn_lstm=False)")

        # Build return dictionary conditionally based on comparison flag
        result = {
            'garch_mae': garch_mae, 
            'garch_mse': garch_mse, 
            'garch_relmae': garch_relmae, 
            'garch_relmse': garch_relmse, 
            'garch_r2': garch_r2
        }
        
        # Only include NN metrics if comparison is enabled
        if self.compare_with_cnn_lstm:
            result.update({
                'nn_mae': nn_mae,
                'nn_mse': nn_mse,
                'nn_relmae': nn_relmae,
                'nn_relmse': nn_relmse,
                'nn_r2': nn_r2
            })
        
        return result
