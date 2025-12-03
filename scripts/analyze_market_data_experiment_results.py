"""
Analysis script for market data experiments with INTC.

This script analyzes the performance of TTSTL experiments on real market data,
comparing different covariate configurations (01, 02, 03, 05, 06) across multiple
date pairs.
"""

# Standard library imports
import json
import logging
import sys
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Local imports
from utils_experiment_analysis import (
    ExperimentMetadata, 
    ExperimentLoader, 
    compare_experiments,
    plot_performance_time_series,
    plot_performance_boxplot,
    analyze_covariate_trends
)
from ttstl.utils.data_downloader import import_to_pandas
from ttstl.utils.data_preprocessor import create_pretrained_scaler

# Import function to load pretrained model setup
import sys
sys.path.insert(0, str(Path(__file__).parent))
from market_data_cnn_lstm_pretraining import load_full_pretrained_setup


def setup_logging():
    """Setup logging for the analysis script."""
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


# Hardcoded experiment directories for market data experiments
# Format: (config_num, pair_num): directory_name
HARDCODED_EXPERIMENTS = {
    ('01', '01'): '2025-11-16_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1031541128_idES010_01_PAIR01_05312024_06032024',
    ('01', '02'): '2025-11-16_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed111354002_idES010_01_PAIR02_06032024_06052024',
    ('01', '03'): '2025-11-16_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed498944675_idES010_01_PAIR03_06052024_06062024',
    ('01', '04'): '2025-11-16_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1365759987_idES010_01_PAIR04_06062024_06102024',
    ('01', '05'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed794588317_idES010_01_PAIR05_06102024_06112024',
    ('01', '06'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1367761692_idES010_01_PAIR06_06112024_06242024',
    ('01', '07'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1667215730_idES010_01_PAIR07_06242024_06252024',
    ('01', '08'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed415630319_idES010_01_PAIR08_06252024_06262024',
    ('01', '09'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1068797469_idES010_01_PAIR09_06262024_06272024',
    ('01', '10'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed7632570_idES010_01_PAIR10_06272024_06282024',
    
    ('02', '01'): '2025-11-16_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1031541128_idES010_02_PAIR01_05312024_06032024',
    ('02', '02'): '2025-11-16_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed111354002_idES010_02_PAIR02_06032024_06052024',
    ('02', '03'): '2025-11-16_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed498944675_idES010_02_PAIR03_06052024_06062024',
    ('02', '04'): '2025-11-16_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1365759987_idES010_02_PAIR04_06062024_06102024',
    ('02', '05'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed794588317_idES010_02_PAIR05_06102024_06112024',
    ('02', '06'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1367761692_idES010_02_PAIR06_06112024_06242024',
    ('02', '07'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1667215730_idES010_02_PAIR07_06242024_06252024',
    ('02', '08'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed415630319_idES010_02_PAIR08_06252024_06262024',
    ('02', '09'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1068797469_idES010_02_PAIR09_06262024_06272024',
    ('02', '10'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed7632570_idES010_02_PAIR10_06272024_06282024',
    
    ('03', '01'): '2025-11-16_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1031541128_idES010_03_PAIR01_05312024_06032024',
    ('03', '02'): '2025-11-16_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed111354002_idES010_03_PAIR02_06032024_06052024',
    ('03', '03'): '2025-11-16_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed498944675_idES010_03_PAIR03_06052024_06062024',
    ('03', '04'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1365759987_idES010_03_PAIR04_06062024_06102024',
    ('03', '05'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed794588317_idES010_03_PAIR05_06102024_06112024',
    ('03', '06'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1367761692_idES010_03_PAIR06_06112024_06242024',
    ('03', '07'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1667215730_idES010_03_PAIR07_06242024_06252024',
    ('03', '08'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed415630319_idES010_03_PAIR08_06252024_06262024',
    ('03', '09'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1068797469_idES010_03_PAIR09_06262024_06272024',
    ('03', '10'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed7632570_idES010_03_PAIR10_06272024_06282024',
    
    ('05', '01'): '2025-11-16_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1031541128_idES010_05_PAIR01_05312024_06032024',
    ('05', '02'): '2025-11-16_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed111354002_idES010_05_PAIR02_06032024_06052024',
    ('05', '03'): '2025-11-16_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed498944675_idES010_05_PAIR03_06052024_06062024',
    ('05', '04'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1365759987_idES010_05_PAIR04_06062024_06102024',
    ('05', '05'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed794588317_idES010_05_PAIR05_06102024_06112024',
    ('05', '06'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1367761692_idES010_05_PAIR06_06112024_06242024',
    ('05', '07'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1667215730_idES010_05_PAIR07_06242024_06252024',
    ('05', '08'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed415630319_idES010_05_PAIR08_06252024_06262024',
    ('05', '09'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1068797469_idES010_05_PAIR09_06262024_06272024',
    ('05', '10'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed7632570_idES010_05_PAIR10_06272024_06282024',
    
    ('06', '01'): '2025-11-16_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1031541128_idES010_06_PAIR01_05312024_06032024',
    ('06', '02'): '2025-11-16_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed111354002_idES010_06_PAIR02_06032024_06052024',
    ('06', '03'): '2025-11-16_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed498944675_idES010_06_PAIR03_06052024_06062024',
    ('06', '04'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1365759987_idES010_06_PAIR04_06062024_06102024',
    ('06', '05'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed794588317_idES010_06_PAIR05_06102024_06112024',
    ('06', '06'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1367761692_idES010_06_PAIR06_06112024_06242024',
    ('06', '07'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1667215730_idES010_06_PAIR07_06242024_06252024',
    ('06', '08'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed415630319_idES010_06_PAIR08_06252024_06262024',
    ('06', '09'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed1068797469_idES010_06_PAIR09_06262024_06272024',
    ('06', '10'): '2025-11-17_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed7632570_idES010_06_PAIR10_06272024_06282024',
}

# Date pair mapping for descriptive labels
PAIR_DATES = {
    '01': ('05312024', '06032024'),
    '02': ('06032024', '06052024'),
    '03': ('06052024', '06062024'),
    '04': ('06062024', '06102024'),
    '05': ('06102024', '06112024'),
    '06': ('06112024', '06242024'),
    '07': ('06242024', '06252024'),
    '08': ('06252024', '06262024'),
    '09': ('06262024', '06272024'),
    '10': ('06272024', '06282024'),
}


def load_market_data_experiments(config_numbers=['01', '02', '03', '05', '06'], 
                                  pairs=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'],
                                  backbone_bar='bb_000'):
    """
    Load performance data for market data experiments.
    
    Args:
        config_numbers: List of configuration numbers to load (e.g., ['01', '02', '03', '05', '06'])
        pairs: List of date pair numbers to load (e.g., ['01', '02', ..., '10'])
        backbone_bar: Which backbone bar to analyze ('bb_000' or 'bb_001')
        
    Returns:
        List of PerformanceData objects
    """
    logger = logging.getLogger(__name__)
    runs_dir = Path('outputs/runs_market_data_experiments')
    
    if not runs_dir.exists():
        logger.error(f"Runs directory not found: {runs_dir}")
        return []
    
    # Build list of experiments based on requested configs and pairs
    all_experiments = []
    for config_num in config_numbers:
        for pair_num in pairs:
            exp_key = (config_num, pair_num)
            if exp_key in HARDCODED_EXPERIMENTS:
                exp_dir = runs_dir / HARDCODED_EXPERIMENTS[exp_key]
                if exp_dir.exists() and (exp_dir / 'config.yaml').exists():
                    all_experiments.append(exp_dir)
                else:
                    logger.warning(f"Hardcoded experiment not found: Config {config_num}, Pair {pair_num} -> {exp_dir}")
            else:
                logger.warning(f"No hardcoded mapping for Config {config_num}, Pair {pair_num}")
    
    logger.info(f"Found {len(all_experiments)} experiments for configs {config_numbers} and pairs {pairs}")
    
    # Load performance data for each experiment (local_adapt mode only)
    perf_data_local = []
    
    for run_dir in all_experiments:
        try:
            metadata = ExperimentMetadata(run_dir=run_dir)
            loader = ExperimentLoader(metadata)
            
            run_id = metadata.get_config_value('run_id')
            
            # Load performance for local_adapt mode
            perf_local = loader.load_performance_data(backbone_bar, mode='local_adapt')
            if perf_local:
                # Add convergence statistics to the performance data object
                failed_count, saddle_point_count = get_convergence_stats(run_dir, backbone_bar)
                perf_local.convergence_failed = failed_count
                perf_local.convergence_saddle_point = saddle_point_count
                
                perf_data_local.append(perf_local)
                logger.info(f"Loaded local_adapt: {run_id} - Failed: {failed_count}, Saddle: {saddle_point_count}")
        
        except Exception as e:
            logger.error(f"Failed to load {run_dir.name}: {e}")
    
    return perf_data_local


def get_convergence_stats(run_dir, backbone_bar):
    """
    Extract convergence statistics from convergence_performance.json.
    
    Args:
        run_dir: Path to experiment run directory
        backbone_bar: Backbone bar identifier (e.g., 'bb_000')
        
    Returns:
        Tuple of (convergence_failed_count, convergence_saddle_point_count)
    """
    convergence_file = run_dir / backbone_bar / 'local_adapt' / 'adapter' / 'convergence_performance.json'
    
    if not convergence_file.exists():
        return 0, 0
    
    try:
        with open(convergence_file, 'r') as f:
            convergence_data = json.load(f)
        
        processing_stats = convergence_data.get('processing_stats', {})
        
        failed_count = 0
        saddle_point_count = 0
        
        for adapter_bar, stats in processing_stats.items():
            status = stats.get('status', '')
            if status == 'failed':
                failed_count += 1
            elif status == 'saddle_point_detected':
                saddle_point_count += 1
        
        return failed_count, saddle_point_count
    
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to read convergence stats from {convergence_file}: {e}")
        return 0, 0


def group_by_config(perf_data_list):
    """Group PerformanceData objects by their configuration number."""
    grouped = dict()
    
    for perf_data in perf_data_list:
        run_id = perf_data.metadata.get_config_value('run_id', default='unknown')
        
        # Extract config number from run_id (e.g., 'ES010_01_PAIR01_...' -> '01')
        if '_' in run_id:
            parts = run_id.split('_')
            if len(parts) >= 2:
                config_num = parts[1]  # Get the config number
                config_name = f'Config {config_num}'
            else:
                config_name = run_id
        else:
            config_name = run_id
        
        if config_name not in grouped:
            grouped[config_name] = []
        
        grouped[config_name].append(perf_data)
    
    return grouped


def performance_comparison():
    """Compare local adapt performance across all configurations and date pairs."""
    logger = logging.getLogger(__name__)
    
    print("Comparing market data experiment configurations performance")
    
    # Create output directory if it doesn't exist
    save_path = Path('outputs/analysis_market_data_experiment_results')
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Define pairs and configs
    pairs = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    config_numbers = ['01', '02', '03', '05', '06']
    backbone_bars = ['bb_000', 'bb_001', 'bb_002']
    
    # Define date labels for each pair
    pair_dates = {
        '01': '06/03',
        '02': '06/05',
        '03': '06/06',
        '04': '06/10',
        '05': '06/11',
        '06': '06/24',
        '07': '06/25',
        '08': '06/26',
        '09': '06/27',
        '10': '06/28'
    }
    
    # Columns to keep from comparison
    cols_to_keep = ['experiment_id', 'overall_rank', 'avg_garch_relmae', 
                   'best_rank', 'worst_rank', 'times_in_top3']
    
    # List to collect all merged dataframes (one per pair)
    all_merged_dfs = []
    
    # Process each pair
    for pair in pairs:
        print(f"  Processing pair {pair}")
        
        # Dictionary to hold comparisons for this pair across all backbones
        pair_comparisons = {}
        
        # Dictionary to hold performance data for each backbone (to access convergence stats)
        perf_data_by_bb = {}
        
        # Load and compare for each backbone bar
        for bb in backbone_bars:
            perf_data = load_market_data_experiments(
                config_numbers=config_numbers,
                pairs=[pair],  # Only this pair
                backbone_bar=bb
            )
            
            if perf_data:
                comparison = compare_experiments(
                    perf_data_list=perf_data,
                    metric='garch_relmae'
                )
                
                if not comparison.empty:
                    pair_comparisons[bb] = comparison
                    perf_data_by_bb[bb] = perf_data
        
        # Merge the three backbone comparisons for this pair
        if len(pair_comparisons) == 3:  # All three backbones loaded successfully
            # Prepare each backbone's data with suffix and add convergence columns
            bb000_subset = pair_comparisons['bb_000'][cols_to_keep].copy()
            # Add convergence columns from PerformanceData attributes
            bb000_subset['convergence_failed_bb000'] = [perf.convergence_failed for perf in perf_data_by_bb['bb_000']]
            bb000_subset['convergence_saddle_point_bb000'] = [perf.convergence_saddle_point for perf in perf_data_by_bb['bb_000']]
            bb000_subset.columns = ['experiment_id'] + [f'{col}_bb000' for col in cols_to_keep[1:]] + ['convergence_failed_bb000', 'convergence_saddle_point_bb000']
            
            bb001_subset = pair_comparisons['bb_001'][cols_to_keep].copy()
            bb001_subset['convergence_failed_bb001'] = [perf.convergence_failed for perf in perf_data_by_bb['bb_001']]
            bb001_subset['convergence_saddle_point_bb001'] = [perf.convergence_saddle_point for perf in perf_data_by_bb['bb_001']]
            bb001_subset.columns = ['experiment_id'] + [f'{col}_bb001' for col in cols_to_keep[1:]] + ['convergence_failed_bb001', 'convergence_saddle_point_bb001']
            
            bb002_subset = pair_comparisons['bb_002'][cols_to_keep].copy()
            bb002_subset['convergence_failed_bb002'] = [perf.convergence_failed for perf in perf_data_by_bb['bb_002']]
            bb002_subset['convergence_saddle_point_bb002'] = [perf.convergence_saddle_point for perf in perf_data_by_bb['bb_002']]
            bb002_subset.columns = ['experiment_id'] + [f'{col}_bb002' for col in cols_to_keep[1:]] + ['convergence_failed_bb002', 'convergence_saddle_point_bb002']
            
            # Merge on experiment_id
            merged_pair = pd.merge(bb000_subset, bb001_subset, on='experiment_id', how='outer')
            merged_pair = pd.merge(merged_pair, bb002_subset, on='experiment_id', how='outer')
            
            # Add a column to identify the pair
            merged_pair.insert(0, 'pair', pair)
            
            # Add to list
            all_merged_dfs.append(merged_pair)
    
    # Stack all pair dataframes vertically
    if all_merged_dfs:
        final_stacked_df = pd.concat(all_merged_dfs, axis=0, ignore_index=True)
        
        # Save the final stacked comparison
        csv_path = save_path / 'comparison_stacked_by_pair.csv'
        final_stacked_df.to_csv(csv_path, index=False)
        print(f"  Saved stacked comparison with {len(final_stacked_df)} rows")
    
    # =========================================================================
    # SEQUENTIAL BOXPLOTS - One per backbone bar
    # =========================================================================
    print("  Creating sequential boxplots by date pair")
    
    # Define colors for each configuration
    config_colors = {
        '01': '#e74c3c',  # Red
        '02': '#3498db',  # Blue
        '03': '#2ecc71',  # Green
        '05': '#9b59b6',  # Purple
        '06': '#1abc9c',  # Turquoise
    }
    
    # Create one plot per backbone bar
    for bb in backbone_bars:
        print(f"    Creating boxplot for {bb}")
        
        # Load all data for this backbone across all pairs
        perf_data_bb = load_market_data_experiments(
            config_numbers=config_numbers,
            pairs=pairs,
            backbone_bar=bb
        )
        
        if not perf_data_bb:
            continue
        
        # Group performance data by pair
        perf_data_by_pair = {}
        for pair in pairs:
            perf_data_by_pair[pair_dates[pair]] = []
        
        for perf_data in perf_data_bb:
            exp_id = perf_data.metadata.get_config_value('run_id', default='unknown')
            
            # Extract pair from experiment ID (e.g., 'ES010_01_PAIR01_...')
            if '_' in exp_id:
                parts = exp_id.split('_')
                if len(parts) >= 3:
                    pair_num = parts[2].replace('PAIR', '')  # e.g., '01' from 'PAIR01'
                    pair_key = pair_dates.get(pair_num)
                    if pair_key and pair_key in perf_data_by_pair:
                        perf_data_by_pair[pair_key].append(perf_data)
        
        # Build color_dict mapping experiment IDs to colors based on config number
        color_dict = {}
        for perf_data in perf_data_bb:
            exp_id = perf_data.metadata.get_config_value('run_id', default='unknown')
            if '_' in exp_id:
                parts = exp_id.split('_')
                if len(parts) >= 2:
                    config_num = parts[1]
                    if config_num in config_colors:
                        color_dict[exp_id] = config_colors[config_num]
        
        # Create legend for configuration colors
        legend_handles = [
            Patch(facecolor=config_colors['01'], alpha=0.7),
            Patch(facecolor=config_colors['02'], alpha=0.7),
            Patch(facecolor=config_colors['03'], alpha=0.7),
            Patch(facecolor=config_colors['05'], alpha=0.7),
            Patch(facecolor=config_colors['06'], alpha=0.7),
        ]
        legend_labels = ['Config 01', 'Config 02', 'Config 03', 'Config 05', 'Config 06']
        legend_config = {
            'handles': legend_handles,
            'labels': legend_labels,
            'loc': 'upper left',
            'bbox_to_anchor': (1.01, 1),
            'fontsize': 11,
            'title': 'Configurations',
            'title_fontsize': 11,
            'framealpha': 0.9,
            'ncol': 1
        }
        
        # Map backbone bar to readable name
        bb_names = {
            'bb_000': 'Backbone Bar 1',
            'bb_001': 'Backbone Bar 2',
            'bb_002': 'Backbone Bar 3'
        }
        bb_name = bb_names.get(bb, bb)
        
        # Create plot using the utility function
        plot_performance_boxplot(
            perf_data_dict=perf_data_by_pair,
            metric='garch_relmae',
            color_dict=color_dict,
            legend_config=legend_config,
            xlabel='Date',
            hline=1.0,
            ylim=(0.975, 1.05),
            figsize=(18, 8),
            save_path=save_path / f'sequential_boxplot_{bb}.png',
            title=f'GARCH RelMAE Performance Across Date Pairs - {bb_name} (Local Adapt Mode)'
        )
        
        # Create plot with CNN-LSTM baseline
        plot_performance_boxplot(
            perf_data_dict=perf_data_by_pair,
            metric='garch_relmae_nn_baseline',
            color_dict=color_dict,
            legend_config=legend_config,
            xlabel='Date',
            hline=1.0,
            ylim=(0.0, 1.05),
            figsize=(18, 8),
            save_path=save_path / f'sequential_boxplot_{bb}_nn_baseline.png',
            title=f'GARCH RelMAE with CNN-LSTM Baseline Across Date Pairs - {bb_name} (Local Adapt Mode)'
        )
    
    print(f"Comparison complete. Produced files are available at {save_path}")


def analyze_market_data_covariates():
    """Analyze covariate dynamics for market data experiments."""
    logger = logging.getLogger(__name__)
    
    print("Analyzing covariate dynamics for market data")
    
    # Create output directory if it doesn't exist
    save_path = Path('outputs/analysis_market_data_experiment_results')
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Configuration and paths
    runs_dir = Path('outputs/runs_market_data_experiments')
    config_num = '01'  # Use configuration 01
    ticker = 'INTC'
    gcs_market_data_folder = 'market_data/'
    
    # Load pretrained scaler and bounds (same as used in experiments)
    pretraining_dir = Path('outputs/cnn_lstm_pretraining_market_data')
    logger.info(f"Loading pretrained scaler and bounds from {pretraining_dir}...")
    
    # Find the .pth file to get the model name
    model_files = list(pretraining_dir.glob("*.pth"))
    if not model_files:
        logger.error(f"No pretrained model files found in {pretraining_dir}")
        return None
    
    model_path = model_files[0]
    model_name = model_path.stem
    
    # Load the pretrained setup (we only need scaler and bounds, but this loads everything)
    _, scaler, lower_bounds, upper_bounds = load_full_pretrained_setup(pretraining_dir, model_name)
    logger.info("Pretrained scaler and bounds loaded successfully")
    
    # Define variables to analyze (including all VIX maturities)
    variables = [
        'micro_return',
        'forward_pass_0',
        'OFI0',
        'vix_1d',
        'vix_9d',
        'vix_30d',
        'vix_3m'
    ]
    
    logger.info(f"Analyzing {len(PAIR_DATES)} date pairs using configuration {config_num}")
    logger.info(f"Variables to analyze: {variables}")
    
    # Store all statistics from all pairs
    all_pair_stats = []
    
    # TODO: Remove this filter when done testing - process all pairs
    # Currently processing only first pair for faster testing
    # test_pairs = {k: v for k, v in list(PAIR_DATES.items())[:1]}
    
    # Process each date pair
    
    # for pair_num, (global_date, local_date) in test_pairs.items(): # TESTING
    for pair_num, (global_date, local_date) in PAIR_DATES.items():
        print(f"\n" + "="*80)
        print(f"Processing Date Pair {pair_num}: {global_date} → {local_date}")
        print("="*80)
        
        # Get experiment directory for this config and pair
        exp_key = (config_num, pair_num)
        if exp_key not in HARDCODED_EXPERIMENTS:
            logger.warning(f"No experiment found for config {config_num}, pair {pair_num}")
            continue
        
        exp_dir = runs_dir / HARDCODED_EXPERIMENTS[exp_key]
        
        if not exp_dir.exists():
            logger.warning(f"Experiment directory not found: {exp_dir}")
            continue
        
        logger.info(f"Using experiment directory: {exp_dir.name}")
        
        # Create ExperimentMetadata from the actual experiment
        metadata = ExperimentMetadata(run_dir=exp_dir)
        
        # Override GCS paths to point to the local adaptation date (not global update)
        metadata.gcs_ed_local_path = f"{gcs_market_data_folder}{local_date}_{ticker}_event_driven_recorded_lob.parquet"
        metadata.gcs_ptd_local_path = f"{gcs_market_data_folder}{local_date}_{ticker}_physical_time_driven_recorded_data.parquet"
        
        logger.info(f"Set GCS paths for local adaptation date {local_date}")
        logger.info(f"  ED: {metadata.gcs_ed_local_path}")
        logger.info(f"  PTD: {metadata.gcs_ptd_local_path}")
        
        # Load event driven and physical time driven data
        loader = ExperimentLoader(metadata)
        ed_df, ptd_df = loader.load_data(mode='local_adapt')
        
        logger.info(f"Loaded event driven data: {ed_df.shape}")
        logger.info(f"Loaded physical time driven data: {ptd_df.shape}")
        
        print(f"\nVariables to analyze: {variables}")
        print("Note: forward_pass_0 will be extracted from the trained CNN-LSTM model")
        print("Using pretrained scaler and bounds (same as in experiments)")
        
        # Analyze each backbone bar for this date pair
        pair_stats = []
        
        for bb in ['bb_000', 'bb_001', 'bb_002']:
            print(f"\n  Analyzing {bb}...")
            
            stats = analyze_covariate_trends(
                experiment=metadata,
                covariate_list=variables,
                backbone_bar=bb,
                mode='local_adapt',
                pretrained_scaler=scaler,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                device='cuda:0',
                batch_size=512,
                compute_ccf=True,
                max_lags=5
            )
            
            # Add metadata columns
            stats['backbone_bar'] = bb
            stats['pair_num'] = pair_num
            stats['global_date'] = global_date
            stats['local_date'] = local_date
            
            pair_stats.append(stats)
            
            print(f"    ✓ Computed statistics for {bb}: {stats.shape[0]} adapter bars")
        
        # Combine all backbone bars for this pair
        pair_combined = pd.concat(pair_stats, ignore_index=True)
        all_pair_stats.append(pair_combined)
        
        print(f"\n  Completed pair {pair_num}: {len(pair_combined)} total rows")
    
    # Combine all pairs
    if all_pair_stats:
        all_stats = pd.concat(all_pair_stats, ignore_index=True)
        
        # Rearrange columns: metadata first, then statistics
        # Drop global_date column as requested
        metadata_cols = ['pair_num', 'local_date', 'backbone_bar', 'adapter_bar']
        stat_cols = [col for col in all_stats.columns if col not in metadata_cols + ['global_date']]
        
        # Reorder columns
        all_stats = all_stats[metadata_cols + stat_cols]
        
        # Save combined statistics to CSV
        csv_path = save_path / 'covariate_statistics_all_pairs_all_backbones.csv'
        all_stats.to_csv(csv_path, index=False)
        
        print(f"\n" + "="*80)
        print(f"Covariate analysis complete!")
        print(f"Combined statistics saved to: {csv_path}")
        # print(f"Total date pairs processed: {len(test_pairs)} (testing mode - only first pair)")
        print(f"Total date pairs available: {len(PAIR_DATES)}")
        print(f"Total rows: {len(all_stats)}")
        print(f"Columns: {list(all_stats.columns)}")
        print("="*80)
        
        return all_stats
    else:
        print("\nNo statistics were computed")
        return None


def plot_covariate_aggregate_measures():
    """
    Create boxplots for covariate aggregate measures.
    
    For each aggregate measure (mean, std, skewness, kurtosis, cv):
        - Create one plot with 3 vertically stacked subplots (one per backbone bar)
        - X-axis: Date pairs
        - Y-axis: Aggregate measure value
        - Each boxplot shows distribution across 12 adapter bars for that date pair
    """
    logger = logging.getLogger(__name__)
    
    print("\nCreating covariate aggregate measure boxplots")
    
    # Load the saved covariate statistics
    save_path = Path('outputs/analysis_market_data_experiment_results')
    csv_path = save_path / 'covariate_statistics_all_pairs_all_backbones.csv'
    
    if not csv_path.exists():
        print(f"Error: Covariate statistics file not found at {csv_path}")
        print("Please run analyze_market_data_covariates() first")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded covariate statistics: {df.shape}")
    
    # Define variables and aggregate measures
    # Note: OFI0 and VIX variables are excluded from all plots (plotted separately due to scale)
    variables_by_measure = {
        'mean': ['micro_return', 'forward_pass_0'],  # Exclude OFI0 and VIX
        'std': ['micro_return', 'forward_pass_0'],   # Exclude OFI0 and VIX
        'skewness': ['micro_return', 'forward_pass_0', 'OFI0'],  # Include OFI0
        'kurtosis': ['micro_return', 'forward_pass_0', 'OFI0'],  # Include OFI0
        'cv': ['micro_return', 'forward_pass_0', 'OFI0']  # Include OFI0
    }
    aggregate_measures = ['mean', 'std', 'skewness', 'kurtosis', 'cv']
    backbone_bars = ['bb_000', 'bb_001', 'bb_002']
    
    # Map backbone bars to readable names
    bb_names = {
        'bb_000': 'Backbone Bar 1',
        'bb_001': 'Backbone Bar 2',
        'bb_002': 'Backbone Bar 3'
    }
    
    # Map aggregate measures to readable names and y-axis labels
    measure_labels = {
        'mean': ('Mean', 'Mean Value'),
        'std': ('Standard Deviation', 'Std Dev'),
        'skewness': ('Skewness', 'Skewness'),
        'kurtosis': ('Kurtosis', 'Kurtosis'),
        'cv': ('Coefficient of Variation', 'CV')
    }
    
    # Date pair labels (for x-axis)
    # Note: pair_num in CSV is stored as string '01', '02', etc.
    date_labels = {
        '01': '06/03',
        '02': '06/05',
        '03': '06/06',
        '04': '06/10',
        '05': '06/11',
        '06': '06/24',
        '07': '06/25',
        '08': '06/26',
        '09': '06/27',
        '10': '06/28'
    }
    
    # Ensure pair_num is string type for consistent lookup
    df['pair_num'] = df['pair_num'].astype(str).str.zfill(2)
    
    # Define y-axis limits for each measure
    ylim_dict = {
        'mean': (-0.0325, 0.0325),
        'std': (0, 0.325),
        'skewness': (-4.25, 6.25),
        'kurtosis': (0, 72.5),
        'cv': (-225, 225)
    }
    
    # Create one plot per aggregate measure
    for measure in aggregate_measures:
        measure_name, ylabel = measure_labels[measure]
        print(f"  Creating boxplots for {measure_name}...")
        
        # Get the variable list for this measure
        variables = variables_by_measure[measure]
        
        # Create figure with 3 vertically stacked subplots
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        fig.suptitle(f'{measure_name} Across Date Pairs and Backbone Bars', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Plot each backbone bar in its own subplot
        for idx, bb in enumerate(backbone_bars):
            ax = axes[idx]
            
            # Filter data for this backbone bar
            bb_df = df[df['backbone_bar'] == bb].copy()
            
            # Prepare data for boxplot: one box per (date_pair, variable) combination
            boxplot_data = []
            boxplot_positions = []
            boxplot_colors = []
            boxplot_labels = []
            
            # Color palette for variables
            var_colors = {
                'micro_return': '#e74c3c',      # Red
                'forward_pass_0': '#3498db',    # Blue
                'OFI0': '#2ecc71',              # Green
                'vix_1d': '#9b59b6',            # Purple
                'vix_9d': '#8e44ad',            # Dark Purple
                'vix_30d': '#f39c12',           # Orange
                'vix_3m': '#e67e22'             # Dark Orange
            }
            
            position = 1
            for pair_num in sorted(df['pair_num'].unique()):
                pair_df = bb_df[bb_df['pair_num'] == pair_num]
                
                for var in variables:
                    col_name = f'{var}_{measure}'
                    if col_name in pair_df.columns:
                        values = pair_df[col_name].dropna().values
                        if len(values) > 0:
                            boxplot_data.append(values)
                            boxplot_positions.append(position)
                            boxplot_colors.append(var_colors[var])
                            
                            # Label only for first pair (for legend)
                            if pair_num == '01':
                                boxplot_labels.append(var)
                            
                            position += 1
                
                # Add spacing between date pairs
                position += 1
            
            # Create boxplot
            bp = ax.boxplot(boxplot_data, positions=boxplot_positions, widths=0.6,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5),
                           medianprops=dict(color='black', linewidth=2))
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], boxplot_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Set y-axis limits
            ymin, ymax = ylim_dict[measure]
            ax.set_ylim(ymin, ymax)
            
            # Add outlier indicators (dots at margin for values outside limits)
            for i, (pos, data, color) in enumerate(zip(boxplot_positions, boxplot_data, boxplot_colors)):
                # Check for outliers above upper limit
                upper_outliers = data[data > ymax]
                if len(upper_outliers) > 0:
                    ax.plot(pos, ymax, marker='D', markersize=8, color=color, 
                           markeredgecolor='black', markeredgewidth=0.5, zorder=10)
                
                # Check for outliers below lower limit
                lower_outliers = data[data < ymin]
                if len(lower_outliers) > 0:
                    ax.plot(pos, ymin, marker='D', markersize=8, color=color,
                           markeredgecolor='black', markeredgewidth=0.5, zorder=10)
            
            # Set up x-axis ticks (centered on each date group)
            pair_positions = []
            pair_labels_list = []
            position = 1
            for pair_num in sorted(df['pair_num'].unique()):
                # Center position for this date pair (middle of 5 variables)
                center_pos = position + 2  # (0 + 1 + 2 + 3 + 4) / 5 = 2
                pair_positions.append(center_pos)
                pair_labels_list.append(date_labels[pair_num])
                position += len(variables) + 1
            
            ax.set_xticks(pair_positions)
            ax.set_xticklabels(pair_labels_list)
            
            # Labels and styling
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(bb_names[bb], fontsize=13, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', labelsize=11)
            
            # Add horizontal line at 0 if appropriate (for mean, skewness)
            if measure in ['mean', 'skewness']:
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Create legend (only on bottom subplot)
        legend_elements = [Patch(facecolor=var_colors[var], alpha=0.7, label=var.replace('_', ' ').title()) 
                          for var in variables]
        axes[-1].legend(handles=legend_elements, loc='upper right', fontsize=11, 
                       ncol=5, framealpha=0.9, title='Variables', title_fontsize=11)
        
        # Common x-axis label
        axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save figure
        save_file = save_path / f'covariate_aggregate_{measure}_boxplots.png'
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {save_file.name}")
    
    print(f"  Boxplot generation complete. Files saved to {save_path}")


def plot_ofi_aggregate_measures_separately():
    """
    Create separate boxplots for OFI mean and std aggregate measures.
    
    OFI has a very different scale compared to other variables, so we plot it separately
    for better visualization. Creates plots for mean and std only (the most important measures).
    """
    logger = logging.getLogger(__name__)
    
    print("\nCreating separate OFI aggregate measure boxplots")
    
    # Load the saved covariate statistics
    save_path = Path('outputs/analysis_market_data_experiment_results')
    csv_path = save_path / 'covariate_statistics_all_pairs_all_backbones.csv'
    
    if not csv_path.exists():
        print(f"Error: Covariate statistics file not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Ensure pair_num is string type
    df['pair_num'] = df['pair_num'].astype(str).str.zfill(2)
    
    backbone_bars = ['bb_000', 'bb_001', 'bb_002']
    
    # Map backbone bars to readable names
    bb_names = {
        'bb_000': 'Backbone Bar 1',
        'bb_001': 'Backbone Bar 2',
        'bb_002': 'Backbone Bar 3'
    }
    
    # Date pair labels (for x-axis)
    date_labels = {
        '01': '06/03',
        '02': '06/05',
        '03': '06/06',
        '04': '06/10',
        '05': '06/11',
        '06': '06/24',
        '07': '06/25',
        '08': '06/26',
        '09': '06/27',
        '10': '06/28'
    }
    
    # Define y-axis limits for each measure
    ofi_ylim_dict = {
        'mean': (-1100, 775),
        'std': (2000, 12250)
    }
    
    # Create plots for OFI mean and std
    for measure in ['mean', 'std']:
        measure_name = 'Mean' if measure == 'mean' else 'Standard Deviation'
        ylabel = 'Mean Value' if measure == 'mean' else 'Std Dev'
        
        print(f"  Creating OFI {measure_name} boxplots...")
        
        # Create figure with 3 vertically stacked subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(f'OFI {measure_name} Across Date Pairs and Backbone Bars', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Plot each backbone bar in its own subplot
        for idx, bb in enumerate(backbone_bars):
            ax = axes[idx]
            
            # Filter data for this backbone bar
            bb_df = df[df['backbone_bar'] == bb].copy()
            
            # Prepare data for boxplot: one box per date pair
            boxplot_data = []
            boxplot_positions = []
            boxplot_labels = []
            
            col_name = f'OFI0_{measure}'
            
            for pair_idx, pair_num in enumerate(sorted(df['pair_num'].unique()), start=1):
                pair_df = bb_df[bb_df['pair_num'] == pair_num]
                
                if col_name in pair_df.columns:
                    values = pair_df[col_name].dropna().values
                    if len(values) > 0:
                        boxplot_data.append(values)
                        boxplot_positions.append(pair_idx)
                        boxplot_labels.append(date_labels[pair_num])
            
            # Create boxplot
            bp = ax.boxplot(boxplot_data, positions=boxplot_positions, widths=0.6,
                           patch_artist=True, showfliers=True,
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5),
                           medianprops=dict(color='black', linewidth=2))
            
            # Color the boxes with OFI green color
            for patch in bp['boxes']:
                patch.set_facecolor('#2ecc71')  # Green
                patch.set_alpha(0.7)
            
            # Set y-axis limits
            ymin, ymax = ofi_ylim_dict[measure]
            ax.set_ylim(ymin, ymax)
            
            # Add outlier indicators (dots at margin for values outside limits)
            for i, (pos, data) in enumerate(zip(boxplot_positions, boxplot_data)):
                # Check for outliers above upper limit
                upper_outliers = data[data > ymax]
                if len(upper_outliers) > 0:
                    ax.plot(pos, ymax, marker='D', markersize=8, color='#2ecc71',
                           markeredgecolor='black', markeredgewidth=0.5, zorder=10)
                
                # Check for outliers below lower limit
                lower_outliers = data[data < ymin]
                if len(lower_outliers) > 0:
                    ax.plot(pos, ymin, marker='D', markersize=8, color='#2ecc71',
                           markeredgecolor='black', markeredgewidth=0.5, zorder=10)
            
            # Labels and styling
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(bb_names[bb], fontsize=13, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', labelsize=11)
            
            # Add horizontal line at 0 if appropriate (for mean)
            if measure == 'mean':
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            # Set x-axis ticks
            ax.set_xticks(boxplot_positions)
            ax.set_xticklabels(boxplot_labels)
        
        # Common x-axis label
        axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save figure
        save_file = save_path / f'ofi_aggregate_{measure}_boxplots.png'
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {save_file.name}")
    
    print(f"  OFI boxplot generation complete. Files saved to {save_path}")


def plot_vix_aggregate_measures_separately():
    """
    Create separate boxplots for all VIX aggregate measures across all maturities.
    
    VIX has a different scale compared to other variables, so we plot it separately
    for better visualization. Creates plots for all aggregate measures (mean, std, 
    skewness, kurtosis, cv) with all four maturities (1d, 9d, 30d, 3m).
    """
    logger = logging.getLogger(__name__)
    
    print("\nCreating separate VIX aggregate measure boxplots")
    
    # Load the saved covariate statistics
    save_path = Path('outputs/analysis_market_data_experiment_results')
    csv_path = save_path / 'covariate_statistics_all_pairs_all_backbones.csv'
    
    if not csv_path.exists():
        print(f"Error: Covariate statistics file not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Ensure pair_num is string type
    df['pair_num'] = df['pair_num'].astype(str).str.zfill(2)
    
    backbone_bars = ['bb_000', 'bb_001', 'bb_002']
    vix_maturities = ['vix_1d', 'vix_9d', 'vix_30d', 'vix_3m']
    aggregate_measures = ['mean', 'std', 'skewness', 'kurtosis', 'cv']
    
    # Map backbone bars to readable names
    bb_names = {
        'bb_000': 'Backbone Bar 1',
        'bb_001': 'Backbone Bar 2',
        'bb_002': 'Backbone Bar 3'
    }
    
    # Map aggregate measures to readable names and y-axis labels
    measure_labels = {
        'mean': ('Mean', 'Mean Value'),
        'std': ('Standard Deviation', 'Std Dev'),
        'skewness': ('Skewness', 'Skewness'),
        'kurtosis': ('Kurtosis', 'Kurtosis'),
        'cv': ('Coefficient of Variation', 'CV')
    }
    
    # Date pair labels (for x-axis)
    date_labels = {
        '01': '06/03',
        '02': '06/05',
        '03': '06/06',
        '04': '06/10',
        '05': '06/11',
        '06': '06/24',
        '07': '06/25',
        '08': '06/26',
        '09': '06/27',
        '10': '06/28'
    }
    
    # Color palette for VIX maturities (matching standard covariate colors)
    vix_colors = {
        'vix_1d': '#e74c3c',    # Red
        'vix_9d': '#3498db',    # Blue
        'vix_30d': '#2ecc71',   # Green
        'vix_3m': '#9b59b6'     # Purple
    }
    
    # Define y-axis limits for each measure
    vix_ylim_dict = {
        'mean': (6, 16),
        'std': (0, 0.4),
        'skewness': (-2, 2),
        'kurtosis': (-2, 6),
        'cv': (0, 0.05)
    }
    
    # Create plots for all VIX aggregate measures
    for measure in aggregate_measures:
        measure_name, ylabel = measure_labels[measure]
        
        print(f"  Creating VIX {measure_name} boxplots...")
        
        # Create figure with 3 vertically stacked subplots
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        fig.suptitle(f'VIX {measure_name} Across Date Pairs and Backbone Bars (All Maturities)', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Plot each backbone bar in its own subplot
        for idx, bb in enumerate(backbone_bars):
            ax = axes[idx]
            
            # Filter data for this backbone bar
            bb_df = df[df['backbone_bar'] == bb].copy()
            
            # Prepare data for boxplot: one box per (date_pair, maturity) combination
            boxplot_data = []
            boxplot_positions = []
            boxplot_colors = []
            
            position = 1
            for pair_num in sorted(df['pair_num'].unique()):
                pair_df = bb_df[bb_df['pair_num'] == pair_num]
                
                for vix_var in vix_maturities:
                    col_name = f'{vix_var}_{measure}'
                    if col_name in pair_df.columns:
                        values = pair_df[col_name].dropna().values
                        if len(values) > 0:
                            boxplot_data.append(values)
                            boxplot_positions.append(position)
                            boxplot_colors.append(vix_colors[vix_var])
                            position += 1
                
                # Add spacing between date pairs
                position += 1
            
            # Create boxplot
            bp = ax.boxplot(boxplot_data, positions=boxplot_positions, widths=0.6,
                           patch_artist=True, showfliers=True,
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5),
                           medianprops=dict(color='black', linewidth=2))
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], boxplot_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Set y-axis limits
            ymin, ymax = vix_ylim_dict[measure]
            ax.set_ylim(ymin, ymax)
            
            # Add outlier indicators (dots at margin for values outside limits)
            for i, (pos, data, color) in enumerate(zip(boxplot_positions, boxplot_data, boxplot_colors)):
                # Check for outliers above upper limit
                upper_outliers = data[data > ymax]
                if len(upper_outliers) > 0:
                    ax.plot(pos, ymax, marker='D', markersize=8, color=color,
                           markeredgecolor='black', markeredgewidth=0.5, zorder=10)
                
                # Check for outliers below lower limit
                lower_outliers = data[data < ymin]
                if len(lower_outliers) > 0:
                    ax.plot(pos, ymin, marker='D', markersize=8, color=color,
                           markeredgecolor='black', markeredgewidth=0.5, zorder=10)
            
            # Set up x-axis ticks (centered on each date group)
            pair_positions = []
            pair_labels_list = []
            position = 1
            for pair_num in sorted(df['pair_num'].unique()):
                # Center position for this date pair (middle of 4 VIX maturities)
                center_pos = position + 1.5  # (0 + 1 + 2 + 3) / 4 = 1.5
                pair_positions.append(center_pos)
                pair_labels_list.append(date_labels[pair_num])
                position += len(vix_maturities) + 1
            
            ax.set_xticks(pair_positions)
            ax.set_xticklabels(pair_labels_list)
            
            # Labels and styling
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(bb_names[bb], fontsize=13, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', labelsize=11)
            
            # Add horizontal line at 0 for certain measures
            if measure in ['mean', 'skewness']:
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Create legend (only on bottom subplot)
        legend_elements = [Patch(facecolor=vix_colors[vix_var], alpha=0.7, 
                                label=vix_var.replace('_', ' ').upper()) 
                          for vix_var in vix_maturities]
        # For mean plot, place legend at bottom right; otherwise upper right
        legend_loc = 'lower right' if measure == 'mean' else 'upper right'
        axes[-1].legend(handles=legend_elements, loc=legend_loc, fontsize=11, 
                       ncol=4, framealpha=0.9, title='VIX Maturities', title_fontsize=11)
        
        # Common x-axis label
        axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save figure
        save_file = save_path / f'vix_aggregate_{measure}_boxplots.png'
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {save_file.name}")
    
    print(f"  VIX boxplot generation complete. Files saved to {save_path}")


def plot_ccf_sequential_boxplots():
    """
    Plot CCF results as sequential boxplots for each covariate.
    
    For each covariate:
        - Create a 10x2 grid of subplots (one per date pair)
        - Each subplot shows boxplots across all lags (-max_lags to +max_lags)
        - Three boxplots per lag (one per backbone), colored red, blue, green
        - Y-axis scale is fixed across all subplots for the same covariate
    """
    logger = logging.getLogger(__name__)
    
    print("\nCreating CCF sequential boxplots")
    
    # Load the saved covariate statistics
    save_path = Path('outputs/analysis_market_data_experiment_results')
    csv_path = save_path / 'covariate_statistics_all_pairs_all_backbones.csv'
    
    if not csv_path.exists():
        print(f"Error: Covariate statistics file not found at {csv_path}")
        print("Please run analyze_market_data_covariates() first")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded covariate statistics: {df.shape}")
    
    # Ensure pair_num is string type for consistent lookup
    df['pair_num'] = df['pair_num'].astype(str).str.zfill(2)
    
    # Identify covariates that have CCF columns
    ccf_columns = [col for col in df.columns if '_ccf_lag_' in col]
    if not ccf_columns:
        print("No CCF columns found in the data. Make sure compute_ccf=True was used.")
        return
    
    # Extract unique covariates with CCF data
    covariates = sorted(set(col.split('_ccf_lag_')[0] for col in ccf_columns))
    print(f"Found CCF data for covariates: {covariates}")
    
    # Exclude VIX variables (not stationary)
    vix_variables = ['vix_1d', 'vix_9d', 'vix_30d', 'vix_3m']
    covariates = [cov for cov in covariates if cov not in vix_variables]
    print(f"Excluding VIX variables (not stationary): {vix_variables}")
    print(f"Plotting CCF for stationary covariates: {covariates}")
    
    # Extract lag range from column names
    lag_values = sorted(set(int(col.split('_ccf_lag_')[1]) for col in ccf_columns))
    print(f"Lag range: {min(lag_values)} to {max(lag_values)}")
    
    # Get unique date pairs (sorted)
    date_pairs = sorted(df['pair_num'].unique())
    n_pairs = len(date_pairs)
    print(f"Number of date pairs: {n_pairs}")
    
    # Date pair labels (local adaptation dates)
    date_labels = {
        '01': '06/03/2024',
        '02': '06/05/2024',
        '03': '06/06/2024',
        '04': '06/10/2024',
        '05': '06/11/2024',
        '06': '06/24/2024',
        '07': '06/25/2024',
        '08': '06/26/2024',
        '09': '06/27/2024',
        '10': '06/28/2024'
    }
    
    # Backbone bars
    backbone_bars = sorted(df['backbone_bar'].unique())
    backbone_colors = {'bb_000': '#e74c3c', 'bb_001': '#3498db', 'bb_002': '#2ecc71'}
    
    # Create one figure per covariate
    for covariate in covariates:
        print(f"\n  Creating CCF boxplots for {covariate}...")
        
        # Get all CCF columns for this covariate
        cov_ccf_cols = [col for col in ccf_columns if col.startswith(f"{covariate}_ccf_lag_")]
        
        # Create figure with 5x2 grid
        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        axes = axes.flatten()
        
        # Calculate global y-limits for this covariate
        all_values = []
        for col in cov_ccf_cols:
            all_values.extend(df[col].dropna().values)
        
        if len(all_values) > 0:
            y_min = min(all_values)
            y_max = max(all_values)
            y_margin = (y_max - y_min) * 0.1
            y_limits = (y_min - y_margin, y_max + y_margin)
        else:
            y_limits = (-1, 1)
        
        # Plot each date pair in a separate subplot
        for idx, pair_num in enumerate(date_pairs):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Filter data for this date pair
            pair_data = df[df['pair_num'] == pair_num]
            
            # Prepare data for boxplot
            # For each lag, we'll have 3 boxplots (one per backbone)
            positions = []
            boxplot_data = []
            colors = []
            
            for lag_idx, lag in enumerate(lag_values):
                col_name = f"{covariate}_ccf_lag_{lag}"
                
                for bb_idx, bb in enumerate(backbone_bars):
                    bb_data = pair_data[pair_data['backbone_bar'] == bb][col_name].dropna().values
                    
                    if len(bb_data) > 0:
                        # Position: lag_idx * 4 + bb_idx (leaving space between lag groups)
                        pos = lag_idx * 4 + bb_idx
                        positions.append(pos)
                        boxplot_data.append(bb_data)
                        colors.append(backbone_colors[bb])
            
            # Create boxplots
            if boxplot_data:
                bp = ax.boxplot(boxplot_data, positions=positions, widths=0.6,
                               patch_artist=True, showfliers=True,
                               boxprops=dict(linewidth=1),
                               medianprops=dict(linewidth=1.5, color='black'),
                               whiskerprops=dict(linewidth=1),
                               capprops=dict(linewidth=1),
                               flierprops=dict(marker='o', markersize=2, alpha=0.3))
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
            
            # Customize subplot
            ax.set_ylim(y_limits)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            
            # Get the local adaptation date for this pair
            local_date = date_labels.get(pair_num, f"Pair {pair_num}")
            ax.set_title(local_date, fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Lag', fontsize=8)
            ax.set_ylabel('CCF', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set x-ticks at lag positions (center of each lag group)
            lag_positions = [i * 4 + 1 for i in range(len(lag_values))]
            ax.set_xticks(lag_positions)
            ax.set_xticklabels(lag_values, fontsize=7)
            
            # Add legend to first subplot only
            if idx == 0:
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#e74c3c', alpha=0.6, label='bb_000'),
                    Patch(facecolor='#3498db', alpha=0.6, label='bb_001'),
                    Patch(facecolor='#2ecc71', alpha=0.6, label='bb_002')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=7)
        
        # Hide unused subplots
        for idx in range(n_pairs, len(axes)):
            axes[idx].axis('off')
        
        # Add overall title
        fig.suptitle(f'CCF Sequential Boxplots: {covariate} vs Micro Return', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.995])
        
        # Save figure
        save_file = save_path / f'ccf_sequential_boxplots_{covariate}.png'
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {save_file.name}")
    
    print(f"\n  CCF boxplot generation complete. Files saved to {save_path}")


def main():
    """Main function - comprehensive analysis."""
    setup_logging()
    
    print("Starting analysis of experimental study with market data (INTC)")
    
    # Performance comparison
    performance_comparison()
    
    # Covariate dynamics analysis with CCF (testing)
    # analyze_market_data_covariates()
    
    # Plot CCF results
    plot_ccf_sequential_boxplots()
    
    # Covariate aggregate measure boxplots
    plot_covariate_aggregate_measures()
    
    # Separate OFI boxplots (different scale)
    plot_ofi_aggregate_measures_separately()
    
    # Separate VIX boxplots (different scale, all maturities)
    plot_vix_aggregate_measures_separately()
    
    print("Concluding analysis of experimental study with market data (INTC)")


if __name__ == "__main__":
    main()
