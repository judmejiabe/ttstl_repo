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
                 parameters : dict = {}):
        self.interface = interface
        self.update_method = update_method
        self.arma_order = arma_order
        self.garch_order = garch_order
        self.dist = dist
        self.model = model
        self.archm = archm
        self.parameters = parameters
        self.spec = None  # To store GARCH specification
        self.logger = logging.getLogger('ttstl.adapter')

    def _update_myopic(self, new_parameters: dict):
        self.parameters = new_parameters

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

    def _update_newton_raphson(self):
        pass

    def _update_fischer_scoring(self):
        pass

    def _update_gradient_descent(self):
        pass
    
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
        self.logger.info(f"ADAPTER BAR {adapter_bar_number} - GARCH MODEL PARAMETERS")
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
        self.logger.info(f"BAR-SPECIFIC ESTIMATES (Bar {adapter_bar_number}):")
        self.logger.info(f"{'Parameter':<20} {'Coefficient':>12} {'Std Error':>12} {'P-Value':>10}")
        self.logger.info("=" * 80)

        for i, (name, coef, se, pval) in enumerate(zip(coef_names, coef_values, se_values, p_values)):
            display_name = mxreg_mapping[name]
            self.logger.info(f"{display_name:<20} {coef:>12.6f} {se:>12.6f} {pval:>10.4f}")
        
        self.logger.info("=" * 80)

        # Print carried estimates (averaged parameters)
        self.logger.info(f"CARRIED ESTIMATES (Bar {adapter_bar_number}):")
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
        result_dict['goodness_of_fit'] = np.array(r_result.rx2('gof'))
        result_dict['ljung_box_tests'] = np.array(r_result.rx2('lb'))
        result_dict['ljung_box_squared_tests'] = np.array(r_result.rx2('lb_sq'))
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

    def augment_data(self,
                     event_driven_adapter_time_bar_data : pd.DataFrame,
                     physical_time_driven_adapter_time_bar_data : pd.DataFrame,
                     scaler : Optional[StandardScaler] = None,
                     target_col_name : str = 'micro_return',
                     quantile : float = 0.005):
        """Augments the physical_time_driven_adapter_time_bar_data with neural covariates."""

        self.logger.info("Augmenting physical time data with neural covariates")

        neural_covariates = self.interface.backbone.extract_neural_covariates(event_driven_adapter_time_bar_data,
                                                                              physical_time_driven_adapter_time_bar_data,
                                                                              quantile=quantile)
        
        neural_covariates['timestamp'] = neural_covariates['timestamp'].astype(float)
        physical_time_driven_adapter_time_bar_data['timestamp'] = \
            physical_time_driven_adapter_time_bar_data['timestamp'].astype(float)

        # Merge neural covariates with physical time data
        augmented_physical_data = pd.merge_asof(
            physical_time_driven_adapter_time_bar_data,
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
            self.logger.info(f"Scaling covariates: {covariate_names}")
            augmented_physical_data[covariate_names] = scaler.transform(augmented_physical_data[covariate_names])

        self.logger.info(f"Merged neural covariates, resulting data shape: {augmented_physical_data.shape}")

        return augmented_physical_data

    def update_adapter(self, 
                       event_driven_adapter_time_bar_data: pd.DataFrame,
                       physical_time_driven_adapter_time_bar_data: pd.DataFrame,
                       scaler : StandardScaler,
                       adapter_bar_number: int,
                       target_col_name: str = 'micro_return',
                       quantile: float = 0.005,
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
        quantile : float, optional
            Winsorization quantile for neural covariate extraction
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
                target_col_name=target_col_name,
                quantile=quantile
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

            self.logger.info(f"GARCH model fitted successfully. Coefficients: {list(new_parameters.keys())}")

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
            else:
                self.logger.info(f"Update method '{self.update_method}' not implemented")
                raise ValueError(f"Update method '{self.update_method}' not implemented")
            
            self.spec = garch_results['spec']  # Store GARCH specification
            
            # Print updated parameters summary
            self._print_parameters_summary(adapter_bar_number, new_parameters, garch_results)
            
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
                 target_col_name: str = 'micro_return',
                 quantile: float = 0.005):
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
            quantile (float): The winsorization quantile for neural covariate extraction.

        Returns:
            pd.DataFrame: A DataFrame containing the mean and volatility forecasts.
        """
        self.logger.info("Starting forecast generation with full data preparation.")

        # Validate prerequisites
        if self.spec is None:
            self.logger.error("GARCH specification (self.spec) is not available. Fit a model first.")
            raise ValueError("GARCH specification (self.spec) is not available. Fit a model first.")
        if not hasattr(self.interface, 'selected_variables') or not self.interface.selected_variables:
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
                target_col_name=target_col_name,
                quantile=quantile
            )

            # Critical. At the start of backbone bars, a NaN might arise in the first row. We drop it for simplicity
            if augmented_df.isnull().any(axis = 1).sum() == 1:
                augmented_df = augmented_df.iloc[1:]
                n_in_sample -= 1

            elif augmented_df.isnull().any(axis=1).sum() > 1:
                self.logger.error("More than one NaN found in augmented data. Please check data integrity.")
                raise ValueError("More than one NaN found in augmented data. Please check data integrity.")

            self.logger.info(f"Augmented combined data shape: {augmented_df.shape}")

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
        in_sample_X = in_sample_augmented[selected_vars]
        
        # Out-of-sample data
        out_of_sample_y = out_of_sample_augmented[target_col_name]
        out_of_sample_X = out_of_sample_augmented[selected_vars]

        # Combine again for the R function call format
        combined_y = pd.concat([in_sample_y, out_of_sample_y])
        combined_X = pd.concat([in_sample_X, out_of_sample_X])

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
                combined_X.values,
                nrow=combined_X.shape[0],
                ncol=combined_X.shape[1]
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
                          target_col_name: str = 'micro_return',
                          quantile: float = 0.005):
        """
        Calculates Mean Absolute Error (MAE) and Mean Squared Error (MSE)
        by leveraging the new forecast method.

        Args:
            in_sample_event_df (pd.DataFrame): In-sample event-driven data.
            in_sample_phys_df (pd.DataFrame): In-sample physical time data.
            out_of_sample_event_df (pd.DataFrame): Out-of-sample event-driven data.
            out_of_sample_phys_df (pd.DataFrame): Out-of-sample physical time data.
            target_col_name (str): The name of the target variable column.
            quantile (float): The winsorization quantile for neural covariate extraction.

        Returns:
            dict: A dictionary containing the calculated 'mae' and 'mse'.
        """
        self.logger.info("Calculating MAE and MSE using the new forecast structure.")

        # Call the new forecast method to get predictions
        forecast_df = self.forecast(
            in_sample_event_df=in_sample_event_df,
            in_sample_phys_df=in_sample_phys_df,
            out_of_sample_event_df=out_of_sample_event_df,
            out_of_sample_phys_df=out_of_sample_phys_df,
            scaler=scaler,
            target_col_name=target_col_name,
            quantile=quantile
        )

        # Ensure alignment between actual and predicted values
        # The forecast_df contains timestamps that we can use for alignment
        # Merge on timestamp to ensure alignment
        forecast_with_actual = pd.merge(
            forecast_df[['timestamp', 'mean']],
            out_of_sample_phys_df[['timestamp', target_col_name]],
            on='timestamp',
            how='inner'
        )
        
        actual_values = forecast_with_actual[target_col_name]
        predicted_values = forecast_with_actual['mean']
        
        self.logger.info(f"Aligned data: {len(actual_values)} observations (forecast: {len(forecast_df)}, original: {len(out_of_sample_phys_df)})")

        # Calculate metrics
        mae = mean_absolute_error(actual_values, predicted_values)
        mse = mean_squared_error(actual_values, predicted_values)

        # Handle benchmark computation with proper type checking and alignment
        try:
            if isinstance(benchmark_forecast, (pd.Series, np.ndarray)):
                # If benchmark is a series, ensure it aligns with actual_values
                if len(benchmark_forecast) != len(actual_values):
                    self.logger.error(f"Benchmark length ({len(benchmark_forecast)}) != actual values length ({len(actual_values)}). Using benchmark mean.")
                    raise ValueError(f"Benchmark length ({len(benchmark_forecast)}) != actual values length ({len(actual_values)}). Using benchmark mean.")
                else:
                    benchmark_value = benchmark_forecast
            else:
                # Scalar benchmark - will broadcast correctly
                benchmark_value = benchmark_forecast
            
            benchmark_errors = np.abs(actual_values - benchmark_value)
            benchmark_mse = np.mean((actual_values - benchmark_value) ** 2)
            
            relmae = mae / np.mean(benchmark_errors) if np.mean(benchmark_errors) != 0 else None
            relmse = mse / benchmark_mse if benchmark_mse != 0 else None
            
        except Exception as e:
            self.logger.warning(f"Benchmark computation failed: {e}. Setting relative metrics to None.")
            relmae = None
            relmse = None
        r2 = 1 - relmse if relmse is not None else None

        self.logger.info(f"Calculated Metrics - MAE: {mae:.6f}, MSE: {mse:.6f}, RelMAE: {relmae:.6f}, RelMSE: {relmse:.6f}, R2: {r2:.6f}")

        return {'mae': mae, 'mse': mse, 'relmae': relmae, 'relmse': relmse, 'r2': r2}
