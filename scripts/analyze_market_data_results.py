# Standard library imports
import json
import logging
import re
from datetime import datetime
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

def format_date_for_display(date_str):
    """
    Convert date string from MMDDYYYY to a more readable format.
    
    Args:
        date_str: Date string in format MMDDYYYY (e.g., "05312024")
        
    Returns:
        Formatted date string (e.g., "May 31")
    """
    try:
        if len(date_str) == 7:  # Handle case like "5312024" -> "05312024"
            date_str = "0" + date_str
        elif len(date_str) == 6:  # Handle case like "632024" -> "06032024"
            date_str = "0" + date_str
        
        month = date_str[:2]
        day = date_str[2:4]
        year = date_str[4:]
        
        # Create datetime object
        dt = datetime.strptime(f"{month}{day}{year}", "%m%d%Y")
        
        # Return formatted string
        return dt.strftime("%b %d")
    except:
        return date_str

def load_covariate_counts(runs_dir):
    """
    Load covariate count data from all experiment runs.
    
    Args:
        runs_dir: Path to runs directory
        
    Returns:
        DataFrame with covariate counts by experiment type, date pair, and backbone bar
    """
    logger = logging.getLogger(__name__)
    runs_path = Path(runs_dir)
    
    covariate_data = []
    
    # Get all run directories
    run_dirs = [d for d in runs_path.iterdir() if d.is_dir()]
    
    for run_dir in run_dirs:
        # Parse directory name
        run_info = parse_run_directory_name(run_dir.name)
        if not run_info:
            continue
        
        # Check all backbone bars for this experiment
        for bb_dir in run_dir.iterdir():
            if bb_dir.is_dir() and bb_dir.name.startswith('bb_'):
                # Extract backbone bar number
                bb_match = re.search(r'bb_(\d+)', bb_dir.name)
                if not bb_match:
                    continue
                bb_number = int(bb_match.group(1))
                
                # Only process global update mode for covariate counts
                selected_vars_file = bb_dir / 'global_update' / 'interface' / 'selected_variables.json'
                
                if selected_vars_file.exists():
                    try:
                        with open(selected_vars_file, 'r') as f:
                            var_data = json.load(f)
                        
                        if 'selected_variables' in var_data:
                            covariate_count = len(var_data['selected_variables'])
                            
                            covariate_data.append({
                                'experiment_type': run_info['experiment_type'],
                                'pair_number': run_info['pair_number'],
                                'global_date': run_info['global_date'],
                                'local_date': run_info['local_date'],
                                'pair_label': run_info['pair_label'],
                                'backbone_bar': bb_number,
                                'covariate_count': covariate_count,
                                'selected_variables': var_data['selected_variables']
                            })
                            
                    except Exception as e:
                        logger.warning(f"Could not load covariate data from {selected_vars_file}: {e}")
    
    covariate_df = pd.DataFrame(covariate_data)
    logger.info(f"Loaded covariate data: {len(covariate_df)} experiment-date-backbone combinations")
    
    return covariate_df

def setup_logging():
    """Setup logging for the analysis script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_run_directory_name(dir_name):
    """
    Parse run directory name to extract experiment information.
    
    Expected format: 2025-09-26_INTC_bdelta7200000000000_adelta600000000000_sdelta2000000000_seed19991810_idMAIN_FULL_PAIR01_05312024_06032024
    
    Args:
        dir_name: Directory name string
        
    Returns:
        Dict with parsed information or None if parsing fails
    """
    # Pattern to match run directory names
    pattern = r'(\d{4}-\d{2}-\d{2})_INTC_.*_id([A-Z_0-9]+)_PAIR(\d{2})_(\d{8})_(\d{8})'
    
    match = re.match(pattern, dir_name)
    if not match:
        return None
    
    run_date, experiment_type, pair_number, global_date, local_date = match.groups()
    
    return {
        'run_date': run_date,
        'experiment_type': experiment_type,
        'pair_number': int(pair_number),
        'global_date': global_date,
        'local_date': local_date,
        'pair_label': f"Pair {pair_number}: {global_date}→{local_date}"
    }

def load_performance_data(runs_dir):
    """
    Load all performance data from TTSTL experiment runs.
    
    Args:
        runs_dir: Path to runs directory
        
    Returns:
        Tuple of (global_update_df, local_adapt_df) containing all performance data
    """
    logger = logging.getLogger(__name__)
    runs_path = Path(runs_dir)
    
    if not runs_path.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    
    global_data = []
    local_data = []
    
    # Get all run directories
    run_dirs = [d for d in runs_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(run_dirs)} run directories")
    
    for run_dir in run_dirs:
        # Parse directory name
        run_info = parse_run_directory_name(run_dir.name)
        if not run_info:
            logger.warning(f"Could not parse directory name: {run_dir.name}")
            continue
        
        logger.info(f"Processing {run_info['experiment_type']} {run_info['pair_label']}")
        
        # Find all backbone bar directories
        bb_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith('bb_')])
        
        for bb_dir in bb_dirs:
            bb_number = int(bb_dir.name.split('_')[1])
            
            # Load global update performance data
            global_perf_file = bb_dir / 'global_update' / 'adapter' / 'performance.parquet'
            if global_perf_file.exists():
                try:
                    df = pd.read_parquet(global_perf_file)
                    df = df.reset_index()  # Reset index to make adapter_bar a column
                    
                    # Add metadata columns
                    df['experiment_type'] = run_info['experiment_type']
                    df['pair_number'] = run_info['pair_number']
                    df['pair_label'] = run_info['pair_label']
                    df['global_date'] = run_info['global_date']
                    df['local_date'] = run_info['local_date']
                    df['backbone_bar'] = bb_number
                    df['mode'] = 'global_update'
                    
                    global_data.append(df)
                    
                except Exception as e:
                    logger.warning(f"Could not load global update data from {global_perf_file}: {e}")
            
            # Load local adapt performance data
            local_perf_file = bb_dir / 'local_adapt' / 'adapter' / 'performance.parquet'
            if local_perf_file.exists():
                try:
                    df = pd.read_parquet(local_perf_file)
                    df = df.reset_index()  # Reset index to make adapter_bar a column
                    
                    # Add metadata columns
                    df['experiment_type'] = run_info['experiment_type']
                    df['pair_number'] = run_info['pair_number']
                    df['pair_label'] = run_info['pair_label']
                    df['global_date'] = run_info['global_date']
                    df['local_date'] = run_info['local_date']
                    df['backbone_bar'] = bb_number
                    df['mode'] = 'local_adapt'
                    
                    local_data.append(df)
                    
                except Exception as e:
                    logger.warning(f"Could not load local adapt data from {local_perf_file}: {e}")
    
    # Combine all data
    global_df = pd.concat(global_data, ignore_index=True) if global_data else pd.DataFrame()
    local_df = pd.concat(local_data, ignore_index=True) if local_data else pd.DataFrame()
    
    logger.info(f"Loaded global update data: {len(global_df)} rows from {len(global_data)} files")
    logger.info(f"Loaded local adapt data: {len(local_df)} rows from {len(local_data)} files")
    
    return global_df, local_df

def create_relmae_stacked_plot(global_df, local_df, output_dir):
    """
    Create stacked panel box plots of RELMAE across experiment types, backbone bars, and dates.
    
    Args:
        global_df: Global update performance data
        local_df: Local adapt performance data
        output_dir: Directory to save the plot
    """
    logger = logging.getLogger(__name__)
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with stacked subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
    fig.suptitle('RELMAE Performance Across All Backbone Bars and Date Pairs', fontsize=16, fontweight='bold')
    
    # Define colors for each experiment type
    exp_colors = {
        'MAIN_FULL': '#1f77b4',        # Blue
        'PARSIMONIOUS': '#ff7f0e',      # Orange
        'BENCHMARK_FP0': '#2ca02c',     # Green
        'NO_COVARIATES': '#d62728'      # Red
    }
    
    # Plot Global Update RELMAE (upper panel) - Box Plot
    if not global_df.empty:
        experiment_types = sorted(global_df['experiment_type'].unique())
        backbone_bars = sorted(global_df['backbone_bar'].unique())
        date_pairs = sorted(global_df['pair_number'].unique())
        
        # Prepare positions for boxplots across dates
        n_exp = len(experiment_types)
        n_bb = len(backbone_bars)
        n_dates = len(date_pairs)
        
        box_data = []
        positions = []
        colors = []
        
        # Create a combined categorical variable for x-axis positioning
        x_pos = 0
        x_ticks = []
        x_labels = []
        
        for date_idx, pair_num in enumerate(date_pairs):
            date_data = global_df[global_df['pair_number'] == pair_num]
            
            # Get date labels for this pair
            if not date_data.empty:
                sample_row = date_data.iloc[0]
                global_date_fmt = format_date_for_display(str(sample_row['global_date']))
                local_date_fmt = format_date_for_display(str(sample_row['local_date']))
                date_label = f"{global_date_fmt}→{local_date_fmt}"
            else:
                date_label = f"Pair {pair_num}"
            
            # Add space between date groups
            if date_idx > 0:
                x_pos += 0.5
            
            date_start_pos = x_pos
            
            for exp_idx, exp_type in enumerate(experiment_types):
                for bb_idx, bb in enumerate(backbone_bars):
                    # Filter data for this combination
                    data = date_data[
                        (date_data['experiment_type'] == exp_type) & 
                        (date_data['backbone_bar'] == bb)
                    ]['relmae'].values
                    
                    if len(data) > 0:
                        box_data.append(data)
                        positions.append(x_pos)
                        colors.append(exp_colors.get(exp_type, 'gray'))
                    
                    x_pos += 0.2
            
            # Add date label at the center of this date group
            date_center = (date_start_pos + x_pos - 0.2) / 2
            x_ticks.append(date_center)
            x_labels.append(date_label)
        
        # Create box plots
        if box_data:
            bp1 = ax1.boxplot(box_data, positions=positions, widths=0.15, patch_artist=True,
                             showmeans=True, meanline=True, 
                             flierprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', 
                                           markersize=3, alpha=0.8))
            
            # Customize box plot colors
            for patch, color in zip(bp1['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Customize other elements
            for element in ['whiskers', 'medians', 'caps']:
                plt.setp(bp1[element], color='black')
            plt.setp(bp1['means'], color='red', linewidth=2)
        
        # Set x-axis labels and ticks
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        
        # Add legend for experiment types only
        legend_elements = []
        for exp_type in experiment_types:
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=exp_colors[exp_type], 
                                               edgecolor='black', alpha=0.7, label=exp_type.replace('_', ' ')))
        ax1.legend(handles=legend_elements, loc='upper right', title='Experiment Types', fontsize=8)
    
    ax1.set_title('Global Update Mode', fontsize=14, fontweight='bold')
    ax1.set_ylabel('RELMAE', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add horizontal line at RELMAE = 1 (no improvement over benchmark)
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.8, linewidth=1)
    
    # Plot Local Adapt RELMAE (lower panel) - Box Plot
    if not local_df.empty:
        experiment_types = sorted(local_df['experiment_type'].unique())
        backbone_bars = sorted(local_df['backbone_bar'].unique())
        date_pairs = sorted(local_df['pair_number'].unique())
        
        box_data = []
        positions = []
        colors = []
        
        # Create a combined categorical variable for x-axis positioning
        x_pos = 0
        x_ticks = []
        x_labels = []
        
        for date_idx, pair_num in enumerate(date_pairs):
            date_data = local_df[local_df['pair_number'] == pair_num]
            
            # Get date labels for this pair
            if not date_data.empty:
                sample_row = date_data.iloc[0]
                global_date_fmt = format_date_for_display(str(sample_row['global_date']))
                local_date_fmt = format_date_for_display(str(sample_row['local_date']))
                date_label = f"{global_date_fmt}→{local_date_fmt}"
            else:
                date_label = f"Pair {pair_num}"
            
            # Add space between date groups
            if date_idx > 0:
                x_pos += 0.5
            
            date_start_pos = x_pos
            
            for exp_idx, exp_type in enumerate(experiment_types):
                for bb_idx, bb in enumerate(backbone_bars):
                    # Filter data for this combination
                    data = date_data[
                        (date_data['experiment_type'] == exp_type) & 
                        (date_data['backbone_bar'] == bb)
                    ]['relmae'].values
                    
                    if len(data) > 0:
                        box_data.append(data)
                        positions.append(x_pos)
                        colors.append(exp_colors.get(exp_type, 'gray'))
                    
                    x_pos += 0.2
            
            # Add date label at the center of this date group
            date_center = (date_start_pos + x_pos - 0.2) / 2
            x_ticks.append(date_center)
            x_labels.append(date_label)
        
        # Create box plots
        if box_data:
            bp2 = ax2.boxplot(box_data, positions=positions, widths=0.15, patch_artist=True,
                             showmeans=True, meanline=True,
                             flierprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', 
                                           markersize=3, alpha=0.8))
            
            # Customize box plot colors
            for patch, color in zip(bp2['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Customize other elements
            for element in ['whiskers', 'medians', 'caps']:
                plt.setp(bp2[element], color='black')
            plt.setp(bp2['means'], color='red', linewidth=2)
        
        # Set x-axis labels and ticks
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        
        # Add legend for experiment types only
        legend_elements = []
        for exp_type in experiment_types:
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=exp_colors[exp_type], 
                                               edgecolor='black', alpha=0.7, label=exp_type.replace('_', ' ')))
        ax2.legend(handles=legend_elements, loc='upper right', title='Experiment Types', fontsize=8)
    
    ax2.set_title('Local Adapt Mode', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RELMAE', fontsize=12)
    ax2.set_xlabel('Date Pairs (Global Update → Local Adapt)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add horizontal line at RELMAE = 1 (no improvement over benchmark)
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.8, linewidth=1)
    
    # Add text annotation explaining the layout
    ax1.text(0.02, 0.98, 'Each date shows: 4 experiment types × 3 backbone bars = 12 boxplots', 
            transform=ax1.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_file = output_path / 'relmae_stacked_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    logger.info(f"RELMAE stacked plot saved to: {plot_file}")
    
    plt.show()
    
    return fig

def create_covariate_count_plot(covariate_df, output_dir):
    """
    Create plot showing number of selected covariates across date pairs and backbone bars.
    
    Args:
        covariate_df: DataFrame with covariate count data
        output_dir: Directory to save the plot
    """
    logger = logging.getLogger(__name__)
    
    if covariate_df.empty:
        logger.warning("No covariate data available for plotting")
        return None
    
    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    fig.suptitle('Number of Selected Covariates Across Date Pairs and Backbone Bars', fontsize=16, fontweight='bold')
    
    # Define colors for each experiment type
    exp_colors = {
        'MAIN_FULL': '#1f77b4',        # Blue
        'PARSIMONIOUS': '#ff7f0e',      # Orange
        'BENCHMARK_FP0': '#2ca02c',     # Green
        'NO_COVARIATES': '#d62728'      # Red
    }
    
    # Define markers for each backbone bar
    bb_markers = {0: 'o', 1: 's', 2: '^'}  # Circle, Square, Triangle
    bb_labels = {0: 'Bar 0', 1: 'Bar 1', 2: 'Bar 2'}
    
    # Plot covariate counts for each experiment type and backbone bar
    for exp_type in covariate_df['experiment_type'].unique():
        for bb_num in sorted(covariate_df['backbone_bar'].unique()):
            exp_bb_data = covariate_df[
                (covariate_df['experiment_type'] == exp_type) & 
                (covariate_df['backbone_bar'] == bb_num)
            ].copy()
            
            if exp_bb_data.empty:
                continue
                
            exp_bb_data = exp_bb_data.sort_values('pair_number')
            
            # Create x-axis offset for each backbone bar
            x_offset = (bb_num - 1) * 0.15  # Spread bars around the center
            
            ax.plot(exp_bb_data['pair_number'] + x_offset, exp_bb_data['covariate_count'], 
                   marker=bb_markers[bb_num], linewidth=1.5, markersize=6,
                   color=exp_colors.get(exp_type, 'gray'), alpha=0.8,
                   label=f"{exp_type.replace('_', ' ')} - {bb_labels[bb_num]}")
    
    ax.set_ylabel('Number of Selected Covariates', fontsize=12)
    ax.set_xlabel('Date Pairs (Global Update → Local Adapt)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Create a custom legend with both experiment types and backbone bars
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    
    # Set x-axis with date labels
    x_ticks = sorted(covariate_df['pair_number'].unique())
    
    # Create date labels for x-axis
    pair_to_dates = covariate_df[['pair_number', 'global_date', 'local_date']].drop_duplicates()
    
    x_labels = []
    for pair_num in x_ticks:
        pair_data = pair_to_dates[pair_to_dates['pair_number'] == pair_num].iloc[0]
        global_date_fmt = format_date_for_display(str(pair_data['global_date']))
        local_date_fmt = format_date_for_display(str(pair_data['local_date']))
        x_labels.append(f"{global_date_fmt}\n→{local_date_fmt}")
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlim(min(x_ticks) - 0.5, max(x_ticks) + 0.5)
    
    # Set y-axis to show integer values
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Add text annotation explaining the backbone bars
    ax.text(0.02, 0.98, 'Each date pair shows 3 backbone bars:\n○ Bar 0  ■ Bar 1  ▲ Bar 2', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_file = output_path / 'covariate_counts_by_date_pair.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    logger.info(f"Covariate count plot saved to: {plot_file}")
    
    plt.show()
    
    return fig

def create_summary_statistics(global_df, local_df, output_dir):
    """
    Create summary statistics for the RELMAE data.
    
    Args:
        global_df: Global update performance data
        local_df: Local adapt performance data
        output_dir: Directory to save the statistics
    """
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Combine data for summary statistics
    global_df_copy = global_df.copy()
    local_df_copy = local_df.copy()
    
    # Calculate summary statistics by experiment type and mode
    summary_stats = []
    
    for mode, df in [('Global Update', global_df_copy), ('Local Adapt', local_df_copy)]:
        if df.empty:
            continue
            
        for exp_type in df['experiment_type'].unique():
            exp_data = df[df['experiment_type'] == exp_type]
            
            stats = {
                'Mode': mode,
                'Experiment_Type': exp_type,
                'Count': len(exp_data),
                'Mean_RELMAE': exp_data['relmae'].mean(),
                'Std_RELMAE': exp_data['relmae'].std(),
                'Min_RELMAE': exp_data['relmae'].min(),
                'Max_RELMAE': exp_data['relmae'].max(),
                'Median_RELMAE': exp_data['relmae'].median(),
                'Q25_RELMAE': exp_data['relmae'].quantile(0.25),
                'Q75_RELMAE': exp_data['relmae'].quantile(0.75),
                'Below_1_Percent': (exp_data['relmae'] < 1.0).mean() * 100
            }
            summary_stats.append(stats)
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_stats)
    
    if not summary_df.empty:
        summary_file = output_path / 'relmae_summary_statistics.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Summary statistics saved to: {summary_file}")
        
        # Display summary
        print("\n" + "="*80)
        print("RELMAE SUMMARY STATISTICS")
        print("="*80)
        print(summary_df.round(4).to_string(index=False))
        print("="*80)
        
        return summary_df
    else:
        logger.warning("No data available for summary statistics")
        return pd.DataFrame()

def create_time_series_plot(global_df, local_df, output_dir):
    """
    Create time series plot showing RELMAE evolution across date pairs.
    
    Args:
        global_df: Global update performance data
        local_df: Local adapt performance data
        output_dir: Directory to save the plot
    """
    logger = logging.getLogger(__name__)
    
    # Calculate mean RELMAE per experiment type and pair number
    global_mean = global_df.groupby(['experiment_type', 'pair_number', 'global_date', 'local_date'])['relmae'].mean().reset_index()
    local_mean = local_df.groupby(['experiment_type', 'pair_number', 'global_date', 'local_date'])['relmae'].mean().reset_index()
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('RELMAE Evolution Across Date Pairs', fontsize=16, fontweight='bold')
    
    # Define colors for each experiment type
    exp_colors = {
        'MAIN_FULL': '#1f77b4',        # Blue
        'PARSIMONIOUS': '#ff7f0e',      # Orange
        'BENCHMARK_FP0': '#2ca02c',     # Green
        'NO_COVARIATES': '#d62728'      # Red
    }
    
    # Plot Global Update time series
    for exp_type in global_mean['experiment_type'].unique():
        exp_data = global_mean[global_mean['experiment_type'] == exp_type]
        ax1.plot(exp_data['pair_number'], exp_data['relmae'], 
                marker='o', linewidth=2, markersize=6,
                color=exp_colors.get(exp_type, 'gray'),
                label=exp_type.replace('_', ' '))
    
    ax1.set_title('Global Update Mode', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean RELMAE', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    
    # Plot Local Adapt time series
    for exp_type in local_mean['experiment_type'].unique():
        exp_data = local_mean[local_mean['experiment_type'] == exp_type]
        ax2.plot(exp_data['pair_number'], exp_data['relmae'], 
                marker='o', linewidth=2, markersize=6,
                color=exp_colors.get(exp_type, 'gray'),
                label=exp_type.replace('_', ' '))
    
    ax2.set_title('Local Adapt Mode', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Mean RELMAE', fontsize=12)
    ax2.set_xlabel('Date Pairs (Global Update → Local Adapt)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    
    # Set x-axis with date labels
    all_pairs = set(global_mean['pair_number'].unique()) | set(local_mean['pair_number'].unique())
    x_ticks = sorted(list(all_pairs))
    
    # Create date labels for x-axis
    if not global_mean.empty:
        pair_to_dates = global_mean[['pair_number', 'global_date', 'local_date']].drop_duplicates()
    else:
        pair_to_dates = local_mean[['pair_number', 'global_date', 'local_date']].drop_duplicates()
    
    x_labels = []
    for pair_num in x_ticks:
        pair_data = pair_to_dates[pair_to_dates['pair_number'] == pair_num].iloc[0]
        global_date_fmt = format_date_for_display(str(pair_data['global_date']))
        local_date_fmt = format_date_for_display(str(pair_data['local_date']))
        x_labels.append(f"{global_date_fmt}\n→{local_date_fmt}")
    
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels, fontsize=10)
    ax2.set_xlim(min(x_ticks) - 0.3, max(x_ticks) + 0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir)
    time_series_file = output_path / 'relmae_time_series.png'
    plt.savefig(time_series_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    logger.info(f"RELMAE time series plot saved to: {time_series_file}")
    
    plt.show()
    
    return fig

def main():
    """Main function to run the market data analysis."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🔍 Starting market data TTSTL results analysis...")
    print("="*80)
    
    # Define paths
    runs_dir = 'runs_market_data_experiments'
    output_dir = 'market_data_analysis_outputs'
    
    try:
        # Load performance data
        logger.info("Loading performance data from all experiments...")
        global_df, local_df = load_performance_data(runs_dir)
        
        # Load covariate count data
        logger.info("Loading covariate count data from all experiments...")
        covariate_df = load_covariate_counts(runs_dir)
        
        if global_df.empty and local_df.empty:
            logger.error("No performance data found!")
            return
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate stacked RELMAE plot
        logger.info("Creating stacked RELMAE plot...")
        create_relmae_stacked_plot(global_df, local_df, output_dir)
        
        # Generate time series plot
        logger.info("Creating RELMAE time series plot...")
        create_time_series_plot(global_df, local_df, output_dir)
        
        # Generate covariate count plot
        logger.info("Creating covariate count plot...")
        create_covariate_count_plot(covariate_df, output_dir)
        
        # Generate summary statistics
        logger.info("Generating summary statistics...")
        create_summary_statistics(global_df, local_df, output_dir)
        
        # Save raw data for further analysis
        combined_df = pd.concat([global_df, local_df], ignore_index=True)
        if not combined_df.empty:
            raw_data_file = Path(output_dir) / 'combined_performance_data.csv'
            combined_df.to_csv(raw_data_file, index=False)
            logger.info(f"Combined raw data saved to: {raw_data_file}")
        
        print("\n" + "="*80)
        print("🎉 Analysis completed successfully!")
        print(f"📊 Results saved to: {output_dir}/")
        print("📈 Generated plots:")
        print("   - relmae_stacked_analysis.png")
        print("   - relmae_time_series.png")
        print("   - covariate_counts_by_date_pair.png")
        print("📋 Generated data files:")
        print("   - relmae_summary_statistics.csv")
        print("   - combined_performance_data.csv")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()