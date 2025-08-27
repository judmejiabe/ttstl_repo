"""
COMPARISON: Committed vs Current Orchestrator.py
============================================

This file summarizes the key differences between the last committed version 
and your current working version of orchestrator.py that could affect 
seed reproducibility in synthetic_study.py

KEY DIFFERENCES FOUND:
=====================

1. NEW IMPORT ADDED:
   Current version adds: from sklearn.preprocessing import StandardScaler
   
2. NEW METHOD: _initialize_scalers()
   - Added call to self._initialize_scalers() in __init__
   - Creates scaler placeholders for each backbone bar
   - New instance attribute: self.scalers = {key: None for key in keys}

3. SIGNIFICANT CHANGES IN _run_adapter_global_update():
   
   A. NEW SCALER CREATION AND FITTING:
      Current version adds:
      ```python
      # Create scaler
      scaler = StandardScaler()
      augmented_df = current_adapter.augment_data(event_df, 
                                                  phys_df.drop(columns=['backbone_bar', 
                                                                        'adapter_bar']),
                                                  target_col_name = self.cfg.target_col_name, 
                                                  quantile = self.cfg.quantile)
      scaler.fit(augmented_df.drop(columns=['timestamp', self.cfg.target_col_name]))
      self.scalers[f"backbone_{self._bb_idx}"] = scaler
      ```
   
   B. SCALER PASSED TO update_adapter():
      Current version adds: scaler=scaler parameter
      
   C. PERFORMANCE METRICS COMMENTED OUT:
      Current version has this section commented out:
      ```python
      # metrics = current_adapter.calculate_performance_metrics(...)
      # metrics['adapter_bar'] = next_adapter_bar
      # performance_metrics.append(metrics)
      ```
      
   D. PERFORMANCE SAVING COMMENTED OUT:
      ```python
      # # Save performance metrics
      # performance_df = pd.DataFrame(performance_metrics)
      # if not performance_df.empty:
      #     performance_df.set_index('adapter_bar', inplace=True)
      #     performance_file = artifacts_dir / "performance.parquet"
      #     performance_df.to_parquet(performance_file)
      #     self.logger.info(f"Saved performance metrics to {performance_file}")
      ```

4. PLOT TITLE CHANGES:
   - Changed 'Adapter Parameter Evolution' to 'Adapter Estimates Evolution'
   - Changed ylabel from 'Parameter' to 'Covariate'

5. LARGE COMMENTED OUT METHOD:
   - The committed version had a large commented-out alternative implementation
     of _run_adapter_global_update() (about 194 lines) that has been removed

POTENTIAL REPRODUCIBILITY IMPACT:
================================

The most significant change that could affect reproducibility is the addition 
of the StandardScaler. This change affects:

1. DATA PREPROCESSING: The scaler transforms the input data, which changes
   the numerical values passed to the neural networks and GARCH models.

2. SCALER FITTING: StandardScaler.fit() can have different behavior based on:
   - Order of operations 
   - Data filtering/preprocessing differences
   - Floating point precision differences

3. ADAPTER UPDATES: The update_adapter() method now receives a scaler parameter,
   which changes how the adapter processes data.

RECOMMENDATIONS FOR DEBUGGING:
=============================

1. Check if your adapter.py has also changed to handle the scaler parameter
2. Verify that the augment_data() method produces consistent results
3. Consider if the scaler fitting order affects reproducibility
4. Check if any randomness in StandardScaler could affect results

The key insight is that your current version fundamentally changes the data 
preprocessing pipeline by adding standardization, which would naturally 
produce different results even with the same seeds.
"""
