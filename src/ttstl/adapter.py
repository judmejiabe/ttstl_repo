import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import os

# Activate pandas to R conversion
pandas2ri.activate()

rugarch = importr('rugarch')

class Adapter:
    def __init__(self, 
                 interface,
                 update_method: str = 'myopic',
                 arma_order : Tuple[int, int] = (1, 1),
                 garch_order: Tuple[int, int] = (1, 1),
                 dist: str = 'norm',
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
    
    def _update_power_law(self):
        pass

    def _update_exponential_smoothing(self):
        pass

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
        print(f"\n{'='*60}")
        print(f"ADAPTER BAR {adapter_bar_number} - GARCH MODEL PARAMETERS")
        print(f"{'='*60}")
        print(f"Update method: {self.update_method}")
        
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
        print(f"\n📊 BAR-SPECIFIC ESTIMATES (Bar {adapter_bar_number}):")
        print(f"{'Parameter':<20} {'Coefficient':>12} {'Std Error':>12} {'P-Value':>10}")
        print("-" * 60)
        
        for i, (name, coef, se, pval) in enumerate(zip(coef_names, coef_values, se_values, p_values)):
            display_name = mxreg_mapping[name]
            print(f"{display_name:<20} {coef:>12.6f} {se:>12.6f} {pval:>10.4f}")
        
        # Print carried estimates (averaged parameters)
        print(f"\n📈 CARRIED ESTIMATES (Bar {adapter_bar_number}):")
        print(f"{'Parameter':<20} {'Coefficient':>12}")
        print("-" * 35)
        
        for name in coef_names:
            display_name = mxreg_mapping[name]
            carried_value = self.parameters.get(name, 0.0)
            print(f"{display_name:<20} {carried_value:>12.6f}")
        
        print(f"{'='*60}")

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

    def update_adapter(self, 
                       event_driven_adapter_time_bar_data: pd.DataFrame,
                       physical_time_driven_adapter_time_bar_data: pd.DataFrame,
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
        
        # Use interface's _prepare_data method to augment data with neural covariates and lagged variables
        print(f"Preparing data for adapter bar {adapter_bar_number}...")
        try:
            augmented_data = self.interface._prepare_data(
                event_driven_backbone_time_bar_data=event_driven_adapter_time_bar_data,
                physical_time_driven_backbone_time_bar_data=physical_time_driven_adapter_time_bar_data,
                quantile=quantile,
                target_col_name=target_col_name,
                n_lags=0  # No lags for adapter update
            )
        except Exception as e:
            raise RuntimeError(f"Failed to prepare data for adapter bar {adapter_bar_number}: {str(e)}")
        
        # Filter data to only the specific adapter bar if needed
        if 'adapter_bar' in augmented_data.columns:
            # Find the adapter bar that corresponds to our data
            unique_bars = augmented_data['adapter_bar'].unique()
            if len(unique_bars) > 1:
                # Use the first adapter bar if multiple exist
                target_adapter_bar = unique_bars[0]
                adapter_bar_data = augmented_data[augmented_data['adapter_bar'] == target_adapter_bar].copy()
                print(f"Filtered to adapter bar {target_adapter_bar}, {len(adapter_bar_data)} observations")
            else:
                adapter_bar_data = augmented_data.copy()
                print(f"Using all data: {len(adapter_bar_data)} observations")
        else:
            adapter_bar_data = augmented_data.copy()
            print(f"No adapter_bar column found, using all data: {len(adapter_bar_data)} observations")
        
        # Extract target variable
        if target_col_name not in adapter_bar_data.columns:
            raise ValueError(f"Target column '{target_col_name}' not found in augmented data")
        
        y = adapter_bar_data[target_col_name]
        
        if len(y) < 10:  # Minimum data requirement for GARCH
            raise ValueError(f"Insufficient data for GARCH fitting: {len(y)} observations. "
                           "Need at least 10 observations.")
        
        # Get selected variables from interface
        selected_vars = self.interface.selected_variables
        lags_prefix = target_col_name + '_lag_'
        # Filter out lag variables from selected variables
        selected_vars = [var for var in selected_vars if not var.startswith(lags_prefix)]
        print(f"Selected variables from interface: {selected_vars}")
        
        # Check which selected variables are available in the augmented data
        missing_vars = [var for var in selected_vars if var not in adapter_bar_data.columns]
        
        if missing_vars:
            raise ValueError(f"Selected variables not found in augmented data: {missing_vars}")
        
        # Prepare external regressors
        X = adapter_bar_data[selected_vars]
        
        if X.isnull().any().any():
            raise ValueError("External regressors contain NaN values. "
                           "Please ensure all selected variables are present and clean.")
                
        if len(y) < 10:
            raise ValueError(f"After removing NaN values, insufficient data: {len(y)} observations")
        
        # Fit GARCH model
        try:
            print(f"Fitting GARCH model with {len(y)} observations and {len(selected_vars)} covariates...")
            print(y.head())
            print(X.head())
            garch_results = self._fit_garch_model(
                y=y.values,
                X=X.values,
                Z=None,  # No variance regressors for now
                out_sample=0
            )
            
            # Extract fitted parameters
            coef_names = garch_results['coefficient_names']
            coef_values = garch_results['coefficients']
            new_parameters = dict(zip(coef_names, coef_values))
            
            print(f"GARCH model fitted successfully. Coefficients: {list(new_parameters.keys())}")
            
            # Update parameters based on update method
            if self.update_method == 'myopic':
                self._update_myopic(new_parameters)
            elif self.update_method == 'simple_average':
                self._update_simple_average(new_parameters, adapter_bar_number)
            else:
                raise ValueError(f"Update method '{self.update_method}' not implemented")
            
            self.spec = garch_results['spec']  # Store GARCH specification
            
            # Print updated parameters summary
            self._print_parameters_summary(adapter_bar_number, new_parameters, garch_results)
            
            # Prepare return dictionary
            return self.parameters

        except Exception as e:
            raise RuntimeError(f"Failed to update adapter for bar {adapter_bar_number}: {str(e)}")

    def forecast(self, 
                 in_sample_target_variable_data : pd.Series, 
                 in_sample_external_regressors : pd.DataFrame,
                 out_sample_target_variable_data : pd.Series,
                 out_sample_external_regressors : pd.DataFrame):
        r_script_path = os.path.join(os.path.dirname(__file__), 'r_scripts', 'run_garch.R')
        ro.r(f'source("{r_script_path}")')
        set_fixed_parameters = ro.globalenv['set_fixed_parameters']

        spec_with_fixed_pars = set_fixed_parameters(self.spec, ro.ListVector(self.parameters))

        combined_target_variable_data = pd.concat([in_sample_target_variable_data, out_sample_target_variable_data])
        combined_external_regressors = pd.concat([in_sample_external_regressors, out_sample_external_regressors])

        external_forecasts_r = ro.ListVector({'mregfor' : ro.r['matrix'](
                                                        combined_external_regressors.values,
                                                        nrow=combined_external_regressors.shape[0],
                                                        ncol=combined_external_regressors.shape[1])})

        garch_forecast = rugarch.ugarchforecast(fitORspec = spec_with_fixed_pars, 
                                                data = combined_target_variable_data.values,
                                                n_ahead = 1,
                                                n_roll = out_sample_target_variable_data.shape[0]-1,
                                                out_sample = out_sample_target_variable_data.shape[0],
                                                external = external_forecasts_r)
        
        print(rugarch.fitted(garch_forecast))

        return rugarch.fitted(garch_forecast)
