"""
Synthetic Study Analysis Script

This script analyzes the results from the synthetic study experiments,
focusing on RelMAE time series visualization across different configurations,
backbone bars, and update modes (global_update vs local_adapt).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def setup_logging():
    """Setup logging for analysis operations."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def find_experiment_runs(runs_dir: Path = Path('runs_synthetic_study')) -> Dict[str, Path]:
    """
    Find all experiment run directories and map them to experiment IDs.
    
    Args:
        runs_dir: Root directory containing experiment runs
        
    Returns:
        Dictionary mapping experiment_id to run directory path
    """
    logger = logging.getLogger(__name__)
    
    if not runs_dir.exists():
        logger.error(f"Runs directory {runs_dir} does not exist")
        return {}
    
    experiment_runs = {}
    
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir():
            # Try to extract experiment ID from directory name
            # Expected format: YYYY-MM-DD_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_id{EXPERIMENT_ID}
            dir_name = run_dir.name
            if '_id' in dir_name:
                experiment_id = dir_name.split('_id')[-1]
                experiment_runs[experiment_id] = run_dir
                logger.debug(f"Found experiment {experiment_id} in {run_dir}")
    
    logger.info(f"Found {len(experiment_runs)} experiment runs")
    return experiment_runs

def load_performance_data(run_dir: Path, backbone_bar: str, mode: str) -> Optional[pd.DataFrame]:
    """
    Load performance data from a specific backbone bar and mode.
    
    Args:
        run_dir: Path to experiment run directory
        backbone_bar: Backbone bar identifier (e.g., 'bb_000', 'bb_001')
        mode: Update mode ('global_update' or 'local_adapt')
        
    Returns:
        DataFrame with performance data or None if not found
    """
    logger = logging.getLogger(__name__)
    
    # Construct path to performance file
    performance_path = run_dir / backbone_bar / mode / "adapter" / "performance.parquet"
    
    if not performance_path.exists():
        logger.warning(f"Performance file not found: {performance_path}")
        return None
    
    try:
        df = pd.read_parquet(performance_path)
        
        # Add metadata columns
        df['experiment_id'] = run_dir.name.split('_id')[-1] if '_id' in run_dir.name else 'unknown'
        df['backbone_bar'] = backbone_bar
        df['mode'] = mode
        df['adapter_bar'] = df.index  # Assuming index corresponds to adapter bars
        df['run_dir'] = str(run_dir)
        df.reset_index(drop=True, inplace=True)
        
        logger.debug(f"Loaded {len(df)} records from {performance_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading {performance_path}: {str(e)}")
        return None

def extract_experiment_info(experiment_id: str) -> Tuple[str, str]:
    """
    Extract update method and covariate config from experiment ID.
    
    Args:
        experiment_id: Experiment identifier (e.g., 'M_01', 'PL3_04')
        
    Returns:
        Tuple of (update_method_key, covariate_key)
    """
    if '_' in experiment_id:
        parts = experiment_id.split('_')
        return parts[0], parts[1]
    return experiment_id, '00'

def get_experiment_description(update_key: str, cov_key: str) -> str:
    """
    Get human-readable description for experiment configuration.
    
    Args:
        update_key: Update method key (M, SA, PL3, ES010)
        cov_key: Covariate configuration key (01-07)
        
    Returns:
        Human-readable description
    """
    update_descriptions = {
        'M': 'Myopic',
        'SA': 'Simple Average', 
        'PL3': 'Power Law (α=3.0)',
        'ES010': 'Exponential Smoothing (α=0.1)'
    }
    
    cov_descriptions = {
        '01': 'Full mode',
        '02': 'Parsimonious mode', 
        '03': 'Manual covariates only',
        '04': 'Neural covariates only',
        '05': 'Only forward_pass_0',
        '06': 'No covariates',
        '07': 'Only itv'
    }
    
    update_desc = update_descriptions.get(update_key, update_key)
    cov_desc = cov_descriptions.get(cov_key, f'Config {cov_key}')
    
    return f"{update_desc} | {cov_desc}"

def collect_all_performance_data(runs_dir: Path = Path('runs_synthetic_study'), 
                                backbone_bars: List[str] = ['bb_000', 'bb_001'],
                                modes: List[str] = ['global_update', 'local_adapt']) -> pd.DataFrame:
    """
    Collect all performance data from experiments.
    
    Args:
        runs_dir: Root directory containing experiment runs
        backbone_bars: List of backbone bars to analyze
        modes: List of modes to analyze
        
    Returns:
        Combined DataFrame with all performance data
    """
    logger = logging.getLogger(__name__)
    logger.info("Collecting performance data from all experiments...")
    
    experiment_runs = find_experiment_runs(runs_dir)
    all_data = []
    
    for experiment_id, run_dir in experiment_runs.items():
        for backbone_bar in backbone_bars:
            for mode in modes:
                df = load_performance_data(run_dir, backbone_bar, mode)
                if df is not None:
                    all_data.append(df)
    
    if not all_data:
        logger.error("No performance data found")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Collected {len(combined_df)} total performance records")
    
    return combined_df

def plot_relmae_time_series(runs_dir: Path = Path('runs_synthetic_study'), 
                           save_path: Optional[Path] = None) -> None:
    """
    Create RelMAE time series plots comparing all experiment configurations.
    
    Creates a 2x2 panel plot showing:
    - Top left: Global Update, bb_000
    - Top right: Global Update, bb_001  
    - Bottom left: Local Adapt, bb_000
    - Bottom right: Local Adapt, bb_001
    
    Args:
        runs_dir: Root directory containing experiment runs
        save_path: Optional path to save the plot
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating RelMAE time series plots...")
    
    # Collect all performance data
    performance_data = collect_all_performance_data(runs_dir)
    
    if performance_data.empty:
        logger.error("No performance data available for plotting")
        return
    
    # Check if RelMAE column exists
    if 'relmae' not in performance_data.columns:
        logger.error("RelMAE column not found in performance data")
        logger.info(f"Available columns: {list(performance_data.columns)}")
        return
    
    # Define the combinations to plot
    combinations = [
        ('global_update', 'bb_000', 'Global Update - BB 000'),
        ('global_update', 'bb_001', 'Global Update - BB 001'),
        ('local_adapt', 'bb_000', 'Local Adapt - BB 000'),
        ('local_adapt', 'bb_001', 'Local Adapt - BB 001')
    ]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Color palette for different experiments
    unique_experiments = performance_data['experiment_id'].unique()
    colors = sns.color_palette("husl", len(unique_experiments))
    color_map = dict(zip(unique_experiments, colors))
    
    for idx, (mode, backbone_bar, title) in enumerate(combinations):
        ax = axes[idx]
        
        # Filter data for current combination
        subset = performance_data[
            (performance_data['mode'] == mode) & 
            (performance_data['backbone_bar'] == backbone_bar)
        ]
        
        if subset.empty:
            ax.text(0.5, 0.5, f'No data for\n{title}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, style='italic')
            ax.set_title(title, fontsize=14, fontweight='bold')
            continue
        
        # Plot each experiment
        for experiment_id in sorted(unique_experiments):
            exp_data = subset[subset['experiment_id'] == experiment_id]
            
            if exp_data.empty:
                continue
            
            # Extract experiment info for legend
            update_key, cov_key = extract_experiment_info(experiment_id)
            exp_description = get_experiment_description(update_key, cov_key)
            
            # Sort by adapter_bar for proper time series
            exp_data = exp_data.sort_values('adapter_bar')
            
            # Plot the time series
            ax.plot(exp_data['adapter_bar'], exp_data['relmae'], 
                   color=color_map[experiment_id], 
                   marker='o', markersize=4, linewidth=2,
                   label=f"{experiment_id}: {exp_description}",
                   alpha=0.8)
        
        # Customize subplot
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Adapter Bar', fontsize=12)
        ax.set_ylabel('RelMAE', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Set reasonable y-axis limits
        if not subset['relmae'].isna().all():
            y_min, y_max = subset['relmae'].min(), subset['relmae'].max()
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # Overall plot formatting
    plt.suptitle('RelMAE Time Series Comparison Across All Experiments', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def create_summary_statistics(runs_dir: Path = Path('runs_synthetic_study')) -> pd.DataFrame:
    """
    Create summary statistics for all experiments.
    
    Args:
        runs_dir: Root directory containing experiment runs
        
    Returns:
        DataFrame with summary statistics
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating summary statistics...")
    
    performance_data = collect_all_performance_data(runs_dir)
    
    if performance_data.empty:
        logger.error("No performance data available for summary")
        return pd.DataFrame()
    
    # Group by experiment and calculate statistics
    summary_stats = []
    
    for experiment_id in performance_data['experiment_id'].unique():
        exp_data = performance_data[performance_data['experiment_id'] == experiment_id]
        
        update_key, cov_key = extract_experiment_info(experiment_id)
        exp_description = get_experiment_description(update_key, cov_key)
        
        # Calculate statistics for each mode and backbone combination
        for mode in ['global_update', 'local_adapt']:
            for backbone_bar in ['bb_000', 'bb_001']:
                subset = exp_data[
                    (exp_data['mode'] == mode) & 
                    (exp_data['backbone_bar'] == backbone_bar)
                ]
                
                if not subset.empty and 'relmae' in subset.columns:
                    stats = {
                        'experiment_id': experiment_id,
                        'update_method': update_key,
                        'covariate_config': cov_key,
                        'description': exp_description,
                        'mode': mode,
                        'backbone_bar': backbone_bar,
                        'mean_relmae': subset['relmae'].mean(),
                        'median_relmae': subset['relmae'].median(),
                        'std_relmae': subset['relmae'].std(),
                        'min_relmae': subset['relmae'].min(),
                        'max_relmae': subset['relmae'].max(),
                        'count': len(subset)
                    }
                    summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    logger.info(f"Created summary statistics for {len(summary_df)} combinations")
    
    return summary_df

def create_adapter_bar_rankings(runs_dir: Path = Path('runs_synthetic_study')) -> Dict[str, pd.DataFrame]:
    """
    Create ranking tables for experiment configurations at each adapter bar.
    
    Creates separate ranking tables for each combination of:
    - Mode: global_update vs local_adapt
    - Backbone bar: bb_000 vs bb_001
    
    Args:
        runs_dir: Root directory containing experiment runs
        
    Returns:
        Dictionary of DataFrames with rankings for each mode/backbone combination
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating adapter bar rankings...")
    
    # Collect all performance data
    performance_data = collect_all_performance_data(runs_dir)
    
    if performance_data.empty:
        logger.error("No performance data available for rankings")
        return {}
    
    # Check if RelMAE column exists
    if 'relmae' not in performance_data.columns:
        logger.error("RelMAE column not found in performance data")
        return {}
    
    ranking_tables = {}
    
    # Define combinations to analyze
    combinations = [
        ('global_update', 'bb_000'),
        ('global_update', 'bb_001'),
        ('local_adapt', 'bb_000'),
        ('local_adapt', 'bb_001')
    ]
    
    for mode, backbone_bar in combinations:
        logger.info(f"Creating rankings for {mode} - {backbone_bar}")
        
        # Filter data for current combination
        subset = performance_data[
            (performance_data['mode'] == mode) & 
            (performance_data['backbone_bar'] == backbone_bar)
        ].copy()
        
        if subset.empty:
            logger.warning(f"No data found for {mode} - {backbone_bar}")
            continue
        
        # Get unique adapter bars
        adapter_bars = sorted(subset['adapter_bar'].unique())
        
        # Create ranking table
        ranking_data = []
        
        for adapter_bar in adapter_bars:
            # Filter data for current adapter bar
            bar_data = subset[subset['adapter_bar'] == adapter_bar].copy()
            
            if bar_data.empty:
                continue
            
            # Sort by RelMAE (ascending - lower is better)
            bar_data = bar_data.sort_values('relmae', ascending=True)
            
            # Add ranking information
            for rank, (_, row) in enumerate(bar_data.iterrows(), 1):
                update_key, cov_key = extract_experiment_info(row['experiment_id'])
                exp_description = get_experiment_description(update_key, cov_key)
                
                ranking_entry = {
                    'adapter_bar': adapter_bar,
                    'rank': rank,
                    'experiment_id': row['experiment_id'],
                    'update_method': update_key,
                    'covariate_config': cov_key,
                    'description': exp_description,
                    'relmae': row['relmae']
                }
                ranking_data.append(ranking_entry)
        
        if ranking_data:
            ranking_df = pd.DataFrame(ranking_data)
            
            # Create pivot table for easy viewing (experiments as columns, adapter_bars as rows)
            pivot_rankings = ranking_df.pivot_table(
                index='adapter_bar', 
                columns='experiment_id', 
                values='rank', 
                fill_value=None
            )
            
            # Create pivot table for RelMAE values
            pivot_relmae = ranking_df.pivot_table(
                index='adapter_bar', 
                columns='experiment_id', 
                values='relmae', 
                fill_value=None
            )
            
            # Store only ranking tables
            combo_key = f"{mode}_{backbone_bar}"
            ranking_tables[f"{combo_key}_ranks"] = pivot_rankings
            
            logger.info(f"Created rankings for {len(adapter_bars)} adapter bars and {len(ranking_df['experiment_id'].unique())} experiments")
    
    return ranking_tables

def save_ranking_tables(ranking_tables: Dict[str, pd.DataFrame], 
                       output_dir: Path = Path('analysis_outputs')) -> None:
    """
    Save ranking tables to CSV files.
    
    Args:
        ranking_tables: Dictionary of ranking DataFrames
        output_dir: Directory to save the files
    """
    logger = logging.getLogger(__name__)
    logger.info("Saving ranking tables...")
    
    output_dir.mkdir(exist_ok=True)
    
    for table_name, df in ranking_tables.items():
        if df.empty:
            continue
        
        # Only save _ranks tables
        if table_name.endswith('_ranks'):
            file_path = output_dir / f"rankings_{table_name}.csv"
            df.to_csv(file_path, index=True)
            logger.info(f"Saved ranking table to {file_path}")

def analyze_ranking_consistency(ranking_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Analyze ranking consistency across adapter bars.
    
    Calculates metrics like average rank, rank variance, and best/worst performance
    for each experiment across all adapter bars.
    
    Args:
        ranking_tables: Dictionary of ranking DataFrames
        
    Returns:
        DataFrame with consistency analysis
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing ranking consistency...")
    
    consistency_data = []
    
    # Analyze each mode/backbone combination
    for table_key in ranking_tables.keys():
        if not table_key.endswith('_ranks'):
            continue
            
        mode_backbone = table_key.replace('_ranks', '')
        rankings_df = ranking_tables[table_key]
        
        if rankings_df.empty:
            continue
        
        # Calculate consistency metrics for each experiment
        for experiment_id in rankings_df.columns:
            ranks = rankings_df[experiment_id].dropna()
            
            if len(ranks) == 0:
                continue
            
            update_key, cov_key = extract_experiment_info(experiment_id)
            exp_description = get_experiment_description(update_key, cov_key)
            
            consistency_entry = {
                'mode_backbone': mode_backbone,
                'experiment_id': experiment_id,
                'update_method': update_key,
                'covariate_config': cov_key,
                'description': exp_description,
                'avg_rank': ranks.mean(),
                'median_rank': ranks.median(),
                'std_rank': ranks.std(),
                'best_rank': ranks.min(),
                'worst_rank': ranks.max(),
                'rank_range': ranks.max() - ranks.min(),
                'num_observations': len(ranks),
                'times_ranked_1st': (ranks == 1).sum(),
                'times_in_top3': (ranks <= 3).sum()
            }
            consistency_data.append(consistency_entry)
    
    consistency_df = pd.DataFrame(consistency_data)
    
    # Add rank of avg_rank within each mode_backbone group
    if not consistency_df.empty:
        consistency_df['avg_rank_rank'] = consistency_df.groupby('mode_backbone')['avg_rank'].rank(method='min')
    
    logger.info(f"Created consistency analysis for {len(consistency_df)} experiment-mode combinations")
    
    return consistency_df

def print_ranking_summary(ranking_tables: Dict[str, pd.DataFrame]) -> None:
    """
    Print a summary of ranking results to console.
    
    Args:
        ranking_tables: Dictionary of ranking DataFrames
    """
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*80)
    print("EXPERIMENT RANKING SUMMARY")
    print("="*80)
    
    for table_key in sorted(ranking_tables.keys()):
        if not table_key.endswith('_ranks'):
            continue
            
        mode_backbone = table_key.replace('_ranks', '')
        rankings_df = ranking_tables[table_key]
        
        if rankings_df.empty:
            continue
        
        print(f"\n{mode_backbone.upper().replace('_', ' ')}:")
        print("-" * 40)
        
        # Calculate average ranks across all adapter bars
        avg_ranks = rankings_df.mean(axis=0, skipna=True).sort_values()
        
        print("Top 5 performers (by average rank):")
        for i, (exp_id, avg_rank) in enumerate(avg_ranks.head(5).items(), 1):
            update_key, cov_key = extract_experiment_info(exp_id)
            exp_description = get_experiment_description(update_key, cov_key)
            print(f"  {i}. {exp_id}: {exp_description} (Avg Rank: {avg_rank:.2f})")
        
        # Show most consistent performers
        rank_stds = rankings_df.std(axis=0, skipna=True).sort_values()
        print(f"\nMost consistent performers (lowest rank variance):")
        for i, (exp_id, rank_std) in enumerate(rank_stds.head(3).items(), 1):
            update_key, cov_key = extract_experiment_info(exp_id)
            exp_description = get_experiment_description(update_key, cov_key)
            avg_rank = avg_ranks[exp_id]
            print(f"  {i}. {exp_id}: {exp_description} (Avg: {avg_rank:.2f}, Std: {rank_std:.2f})")

def main():
    """Main analysis function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
def main():
    """Main analysis function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting synthetic study analysis...")
    
    # Define paths
    runs_dir = Path('runs_synthetic_study')
    output_dir = Path('analysis_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Create RelMAE time series plots
    plot_save_path = output_dir / 'relmae_time_series_comparison.png'
    plot_relmae_time_series(runs_dir, plot_save_path)
    
    # Create adapter bar rankings
    logger.info("\nCreating adapter bar rankings...")
    ranking_tables = create_adapter_bar_rankings(runs_dir)
    
    if ranking_tables:
        # Save ranking tables
        save_ranking_tables(ranking_tables, output_dir)
        
        # Create ranking consistency analysis
        consistency_analysis = analyze_ranking_consistency(ranking_tables)
        if not consistency_analysis.empty:
            consistency_path = output_dir / 'ranking_consistency_analysis.csv'
            consistency_analysis.to_csv(consistency_path, index=False)
            logger.info(f"Ranking consistency analysis saved to {consistency_path}")
        
        # Print ranking summary to console
        print_ranking_summary(ranking_tables)
        
        # Additional insights
        logger.info("\nAdditional insights:")
        for table_key in ranking_tables.keys():
            if table_key.endswith('_ranks'):
                mode_backbone = table_key.replace('_ranks', '')
                rankings_df = ranking_tables[table_key]
                
                if not rankings_df.empty:
                    total_comparisons = rankings_df.count().sum()
                    logger.info(f"{mode_backbone}: {len(rankings_df)} adapter bars, {len(rankings_df.columns)} experiments, {total_comparisons} total rankings")
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()