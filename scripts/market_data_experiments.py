# Standard library imports
import copy
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import torch

# Local imports
from ttstl import orchestrator
from ttstl.orchestrator import Orchestrator, OrchestratorConfig
from ttstl.utils.data_downloader import import_to_pandas
from ttstl.utils.experiment_utils import (
    setup_experiment_logging,
    set_trained_model_to_orchestrator,
    log_orchestrator_config,
    get_base_config,
    get_experiment_configs,
    create_experiment_config,
    load_date_data,
    load_pretrained_model_for_experiments
)
















def run_single_date_pair_experiment(config, global_date, local_date, runs_dir, base_orchestrator=None, previous_orchestrator=None, trained_model=None, past_returns=False):
    """
    Run a single date pair experiment with the given configuration.
    
    Args:
        config: Orchestrator configuration for this specific date pair
        global_date: Date string for global update (e.g., "05312024")
        local_date: Date string for local adaptation (e.g., "06032024")
        runs_dir: Directory to save results
        base_orchestrator: Orchestrator to import backbone from (for non-main configs)
        previous_orchestrator: Previous orchestrator from same config type (for state transfer)
        trained_model: Pretrained CNN-LSTM model (only for first main config)
        past_returns: If True, use concurrent returns; if False, use future returns
        
    Returns:
        Orchestrator instance if successful, None if failed
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running single date pair experiment: {config.run_id}")
    logger.info(f"Date pair: {global_date} → {local_date}")
    experiment_type = "CONCURRENT RETURNS" if past_returns else "FUTURE RETURNS"
    logger.info(f"Using {experiment_type}")
    
    try:
        # Create orchestrator for this specific date pair
        orchestrator = Orchestrator(runs_dir, config)
        
        # Handle orchestrator state initialization
        if base_orchestrator is not None:
            # For non-main configurations: copy from base orchestrator (from main experiment)
            logger.info(f"Copying orchestrator state from main to {config.run_id}")
            logger.info("This allows full retraining capability while preserving learned state")
            
            # Copy the trained models from base orchestrator
            orchestrator.import_backbone_models_from_orchestrator(base_orchestrator)
            
            
        elif previous_orchestrator is not None:
            # For subsequent date pairs within same experiment: copy from previous orchestrator
            logger.info(f"Copying orchestrator state from previous date pair to {config.run_id}")
            logger.info("This enables sequential learning: previous knowledge + new data training")

            # Copy the trained models from previous orchestrator and KEEP training capability
            orchestrator.import_backbone_models_from_orchestrator(previous_orchestrator, keep_epochs=True)
            
        elif trained_model is not None:
            # For the first date pair of main configuration: use pretrained model
            logger.info(f"Setting initial pretrained CNN-LSTM model to orchestrator {config.run_id}")
            logger.info("This provides the starting point for the main experiment's training sequence")
            set_trained_model_to_orchestrator(orchestrator, trained_model)
        
        # Log configuration for this date pair
        logger.info(f"Configuration for {config.run_id}:")
        logger.info(f"  - Date pair: {global_date} → {local_date}")
        logger.info(f"  - Epochs: {config.epochs}")
        logger.info(f"  - Parsimonious mode: {config.parsimonious_mode}")
        logger.info(f"  - Directory: {orchestrator.run_dir}")
        
        # Load data for global update date
        logger.info(f"Loading data for GLOBAL UPDATE: {global_date}")
        global_ed_df, global_ptd_df = load_date_data(global_date, past_returns=past_returns)
        
        if global_ed_df is None or global_ptd_df is None:
            logger.error(f"Failed to load global update data for {global_date}")
            return None
        
        # Load data for local adapt date  
        logger.info(f"Loading data for LOCAL ADAPT: {local_date}")
        local_ed_df, local_ptd_df = load_date_data(local_date, past_returns=past_returns)
        
        if local_ed_df is None or local_ptd_df is None:
            logger.error(f"Failed to load local adapt data for {local_date}")
            return None
        
        # Run global update (this will retrain the neural network if epochs > 0)
        logger.info(f"Running GLOBAL UPDATE for {config.run_id} on {global_date}")
        logger.info(f"Neural network retraining: {'ENABLED' if config.epochs > 0 else 'DISABLED'}")
        try:
            global_result = orchestrator.global_update(global_ed_df, global_ptd_df)
            logger.info(f"Global update completed for {config.run_id} on {global_date}")
        except Exception as e:
            logger.error(f"Failed global update for {config.run_id} on {global_date}: {str(e)}")
            return None
        
        # Run local adaptation
        logger.info(f"Running LOCAL ADAPTATION for {config.run_id} on {local_date}")
        try:
            local_result = orchestrator.local_adapt(local_ed_df, local_ptd_df)
            logger.info(f"Local adaptation completed for {config.run_id} on {local_date}")
            
            logger.info(f"Successfully completed {config.run_id}")
            logger.info(f"Results saved to directory: {orchestrator.run_dir}")
            return orchestrator
            
        except Exception as e:
            logger.error(f"Failed local adaptation for {config.run_id} on {local_date}: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to create or run experiment {config.run_id}: {str(e)}")
        logger.exception("Full traceback:")
        return None

def run_independent_ttstl_experiments(base_config, experiment_configs, runs_dir, trained_model, past_returns=False, seed=19991810):
    """
    Run TTSTL experiments date-pair by date-pair following synthetic study design.
    For each date pair: run config 01 (trains backbone), then other configurations import it.
    
    Args:
        base_config: Base orchestrator configuration
        experiment_configs: Dictionary of experiment configurations
        runs_dir: Directory to save results
        trained_model: Pretrained CNN-LSTM model (used as starting point for config 01)
        past_returns: If True, use concurrent returns; if False, use future returns
        seed: Base random seed for generating date-pair-specific seeds
    """
    logger = logging.getLogger(__name__)
    experiment_type = "CONCURRENT RETURNS" if past_returns else "FUTURE RETURNS"
    logger.info(f"Starting date-pair-wise TTSTL experiments with {experiment_type}")
    logger.info("Based on synthetic study: ES010 (alpha=0.1), MA(1), configs 01, 02, 03, 05, 06")
    
    # Define the sequence of date pairs for experiments
    experiment_pairs = [
        ("05312024", "06032024"),  # May 31st (Global Update) & June 3rd (Local Adapt)
        ("06032024", "06052024"),  # June 3rd (Global Update) & June 5th (Local Adapt)
        ("06052024", "06062024"),  # June 5th (Global Update) & June 6th (Local Adapt)
        ("06062024", "06102024"),  # June 6th (Global Update) & June 10th (Local Adapt)
        ("06102024", "06112024"),  # June 10th (Global Update) & June 11th (Local Adapt)
        ("06112024", "06242024"),  # June 11th (Global Update) & June 24th (Local Adapt)
        ("06242024", "06252024"),  # June 24th (Global Update) & June 25th (Local Adapt)
        ("06252024", "06262024"),  # June 25th (Global Update) & June 26th (Local Adapt)
        ("06262024", "06272024"),  # June 26th (Global Update) & June 27th (Local Adapt)
        ("06272024", "06282024"),  # June 27th (Global Update) & June 28th (Local Adapt)
    ]
    
    # Prepare experiment configurations (matching synthetic study order: 01, 02, 03, 04, 05, 06)
    config_01 = create_experiment_config(base_config, experiment_configs['01'])
    config_02 = create_experiment_config(base_config, experiment_configs['02'])
    config_03 = create_experiment_config(base_config, experiment_configs['03'])
    config_04 = create_experiment_config(base_config, experiment_configs['04'])
    config_05 = create_experiment_config(base_config, experiment_configs['05'])
    config_06 = create_experiment_config(base_config, experiment_configs['06'])
    logger.info(f"Will process {len(experiment_pairs)} date pairs")
    logger.info("SEQUENTIAL LEARNING DESIGN:")
    logger.info("  - Date pair 1: Start with pretrained model → train 10 epochs")
    logger.info("  - Date pair 2: Start with date pair 1 results → train 10 epochs") 
    logger.info("  - Date pair 3: Start with date pair 2 results → train 10 epochs")
    logger.info("  - And so on... (progressive learning across time)")
    logger.info("For each date pair: Config 01 (train backbone) → Configs 02, 03, 04, 05, 06 (import backbone)")
    
    # Set random seed for reproducible random seed generation
    np.random.seed(seed)
    
    # Track orchestrators for state transfer between date pairs
    previous_01_orchestrator = None
    
    # Counters for summary
    total_experiments = len(experiment_pairs) * 6  # 6 configs per date pair
    successful_experiments = 0
    failed_experiments = 0
    
    # Process each date pair
    for pair_idx, (global_date, local_date) in enumerate(experiment_pairs):
        logger.info(f"\n{'='*100}")
        logger.info(f"PROCESSING DATE PAIR {pair_idx+1}/{len(experiment_pairs)}: {global_date} → {local_date}")
        logger.info(f"{'='*100}")
        
        # Step 1: Run CONFIG 01 for this date pair (trains backbone)
        logger.info(f"\n--- STEP 1: Running CONFIG 01 for pair {pair_idx+1} ---")
        
        # Generate random seed for this date pair to vary validation split
        # This same seed will be used for all configurations in this date pair
        date_pair_seed = np.random.randint(0, 2**31 - 1)
        logger.info(f"Using randomly generated seed {date_pair_seed} for ALL configurations in date pair {pair_idx+1}")
        
        config_01_pair = OrchestratorConfig(**vars(config_01))
        config_01_pair.run_id = f"{config_01.run_id}_PAIR{pair_idx+1:02d}_{global_date}_{local_date}"
        config_01_pair.seed = date_pair_seed
        
        orchestrator_01 = run_single_date_pair_experiment(
            config_01_pair, global_date, local_date, runs_dir,
            base_orchestrator=None,
            previous_orchestrator=previous_01_orchestrator,
            trained_model=trained_model if pair_idx == 0 else None,
            past_returns=past_returns
        )
        
        if orchestrator_01 is not None:
            successful_experiments += 1
            previous_01_orchestrator = orchestrator_01
            logger.info(f"✓ CONFIG 01 completed for pair {pair_idx+1}")
        else:
            failed_experiments += 1
            logger.error(f"✗ CONFIG 01 failed for pair {pair_idx+1}")
            continue
        
        # Step 2: Run CONFIG 02 for this date pair
        logger.info(f"\n--- STEP 2: Running CONFIG 02 for pair {pair_idx+1} ---")
        
        config_02_pair = OrchestratorConfig(**vars(config_02))
        config_02_pair.run_id = f"{config_02.run_id}_PAIR{pair_idx+1:02d}_{global_date}_{local_date}"
        config_02_pair.seed = date_pair_seed  # Use same seed as config 01
        
        orchestrator_02 = run_single_date_pair_experiment(
            config_02_pair, global_date, local_date, runs_dir,
            base_orchestrator=orchestrator_01,
            previous_orchestrator=None,
            trained_model=None,
            past_returns=past_returns
        )
        
        if orchestrator_02 is not None:
            successful_experiments += 1
            logger.info(f"✓ CONFIG 02 completed for pair {pair_idx+1}")
        else:
            failed_experiments += 1
            logger.error(f"✗ CONFIG 02 failed for pair {pair_idx+1}")
        
        # Step 3: Run CONFIG 03 for this date pair
        logger.info(f"\n--- STEP 3: Running CONFIG 03 for pair {pair_idx+1} ---")
        
        config_03_pair = OrchestratorConfig(**vars(config_03))
        config_03_pair.run_id = f"{config_03.run_id}_PAIR{pair_idx+1:02d}_{global_date}_{local_date}"
        config_03_pair.seed = date_pair_seed  # Use same seed as config 01
        
        orchestrator_03 = run_single_date_pair_experiment(
            config_03_pair, global_date, local_date, runs_dir,
            base_orchestrator=orchestrator_01,
            previous_orchestrator=None,
            trained_model=None,
            past_returns=past_returns
        )
        
        if orchestrator_03 is not None:
            successful_experiments += 1
            logger.info(f"✓ CONFIG 03 completed for pair {pair_idx+1}")
        else:
            failed_experiments += 1
            logger.error(f"✗ CONFIG 03 failed for pair {pair_idx+1}")
        
        # Step 4: Run CONFIG 04 for this date pair
        logger.info(f"\n--- STEP 4: Running CONFIG 04 for pair {pair_idx+1} ---")
        
        config_04_pair = OrchestratorConfig(**vars(config_04))
        config_04_pair.run_id = f"{config_04.run_id}_PAIR{pair_idx+1:02d}_{global_date}_{local_date}"
        config_04_pair.seed = date_pair_seed  # Use same seed as config 01
        
        orchestrator_04 = run_single_date_pair_experiment(
            config_04_pair, global_date, local_date, runs_dir,
            base_orchestrator=orchestrator_01,
            previous_orchestrator=None,
            trained_model=None,
            past_returns=past_returns
        )
        
        if orchestrator_04 is not None:
            successful_experiments += 1
            logger.info(f"✓ CONFIG 04 completed for pair {pair_idx+1}")
        else:
            failed_experiments += 1
            logger.error(f"✗ CONFIG 04 failed for pair {pair_idx+1}")
        
        # Step 6: Run CONFIG 06 for this date pair
        logger.info(f"\n--- STEP 6: Running CONFIG 06 for pair {pair_idx+1} ---")
        
        config_05_pair = OrchestratorConfig(**vars(config_05))
        config_05_pair.run_id = f"{config_05.run_id}_PAIR{pair_idx+1:02d}_{global_date}_{local_date}"
        config_05_pair.seed = date_pair_seed  # Use same seed as config 01
        
        orchestrator_05 = run_single_date_pair_experiment(
            config_05_pair, global_date, local_date, runs_dir,
            base_orchestrator=orchestrator_01,
            previous_orchestrator=None,
            trained_model=None,
            past_returns=past_returns
        )
        
        if orchestrator_05 is not None:
            successful_experiments += 1
            logger.info(f"✓ CONFIG 05 completed for pair {pair_idx+1}")
        else:
            failed_experiments += 1
            logger.error(f"✗ CONFIG 05 failed for pair {pair_idx+1}")
        
        # Step 5: Run CONFIG 06 for this date pair
        logger.info(f"\n--- STEP 5: Running CONFIG 06 for pair {pair_idx+1} ---")
        
        config_06_pair = OrchestratorConfig(**vars(config_06))
        config_06_pair.run_id = f"{config_06.run_id}_PAIR{pair_idx+1:02d}_{global_date}_{local_date}"
        config_06_pair.seed = date_pair_seed  # Use same seed as config 01
        
        orchestrator_06 = run_single_date_pair_experiment(
            config_06_pair, global_date, local_date, runs_dir,
            base_orchestrator=orchestrator_01,
            previous_orchestrator=None,
            trained_model=None,
            past_returns=past_returns
        )
        
        if orchestrator_06 is not None:
            successful_experiments += 1
            logger.info(f"✓ CONFIG 06 completed for pair {pair_idx+1}")
        else:
            failed_experiments += 1
            logger.error(f"✗ CONFIG 06 failed for pair {pair_idx+1}")
        
        logger.info(f"\nCompleted all configurations for date pair {pair_idx+1}: {global_date} → {local_date}")
    
    # Final summary
    logger.info("=" * 100)
    logger.info("DATE-PAIR-WISE TTSTL EXPERIMENTS COMPLETED")
    logger.info("=" * 100)
    logger.info(f"Total date pairs processed: {len(experiment_pairs)}")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Successful: {successful_experiments}")
    logger.info(f"Failed: {failed_experiments}")
    logger.info(f"Success rate: {successful_experiments/total_experiments*100:.1f}%")
    logger.info(f"Results saved to: {runs_dir}")
    logger.info("=" * 100)



def main(past_returns=False):
    """
    Main function to orchestrate the market data TTSTL experiments.
    
    This script focuses only on running TTSTL experiments with different configurations
    using a pretrained CNN-LSTM model. The model should be pretrained using the 
    cnn_lstm_pretraining.py script first.
    
    Args:
        past_returns: If True, run experiments with concurrent returns; if False, use future returns
    """
    experiment_type = "CONCURRENT RETURNS" if past_returns else "FUTURE RETURNS"
    print(f"🚀 Starting market data TTSTL experiments with {experiment_type}...")
    print("="*100)

    # Set global random seeds for reproducibility
    SEED = 19991810  # Same seed as in config
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    # PyTorch reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    setup_experiment_logging("market_data_experiments", past_returns)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting market data TTSTL experiments for INTC with {experiment_type}")
    logger.info(f"Global random seed set to: {SEED}")

    # Specify the pretrained model directory
    pretraining_dir = Path('outputs/cnn_lstm_pretraining_market_data')
    
    # Load pretrained CNN-LSTM model and preprocessing components
    logger.info(f"Loading pretrained CNN-LSTM model and preprocessing components from {pretraining_dir}...")
    model_data = load_pretrained_model_for_experiments(pretraining_dir)
    
    if model_data is None:
        logger.error(f"Failed to load pretrained model and preprocessing components from {pretraining_dir}.")
        return
    
    trained_model, scaler, lower_bounds, upper_bounds = model_data
    logger.info(f"Pretrained CNN-LSTM model and preprocessing components loaded successfully for {experiment_type}")
    
    # Create runs directory with experiment type suffix
    experiment_suffix = "_concurrent" if past_returns else ""
    runs_dir = Path(f'outputs/runs_market_data_experiments{experiment_suffix}')
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Get base configuration and experiment configurations
    base_config = get_base_config()
    
    # Update base configuration with experiment type and preprocessing components
    base_config.past_returns = past_returns  # Set the experiment type
    base_config.pretrained_scaler = scaler
    base_config.lower_bounds = lower_bounds
    base_config.upper_bounds = upper_bounds
    
    experiment_configs = get_experiment_configs()

    logger.info("=" * 80)
    logger.info(f"EXPERIMENT SUMMARY - {experiment_type}")
    logger.info("Based on synthetic study results:")
    logger.info("  - Update method: Exponential Smoothing (alpha=0.1)")
    logger.info("  - ARMA order: (0, 1)")
    logger.info("  - Covariate configurations: 01, 02, 03, 04, 05, 06")
    logger.info("=" * 80)
    for exp_key, exp_config in experiment_configs.items():
        logger.info(f"CONFIG {exp_key}: {exp_config['description']}")
        logger.info(f"  - Run ID: {exp_config['run_id']}")
        logger.info(f"  - Parsimonious mode: {exp_config['parsimonious_mode']}")
        logger.info(f"  - User defined variables: {exp_config['user_defined_variables']}")
        logger.info(f"  - Epochs: {exp_config['epochs']}")
        logger.info(f"  - ARMA order: {exp_config['arma_order']}")
    logger.info("=" * 80)

    # Run experiments independently with separate directories
    run_independent_ttstl_experiments(
        base_config, experiment_configs, runs_dir, 
        trained_model, past_returns, seed=SEED
    )
    
    logger.info(f"All TTSTL experiments with {experiment_type} completed successfully!")
    return

if __name__ == "__main__":
    main()