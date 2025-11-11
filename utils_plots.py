
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter
from typing import Callable, Dict
import re

def analyze_channel_by_file(csv_path, output_folder=None, channel='FL8-A'):
    """
    Read processed CSV data, calculate channel mean values grouped by file name and generate bar charts

    Parameters:
    csv_path (str): Path to the CSV file containing merged processed data
    output_folder (str, optional): Output folder for charts, if not specified, charts will only be displayed
    
    Returns:
    pd.DataFrame: DataFrame containing channel mean, standard deviation, and sample size for each file
    """
    # Set simple style
    plt.style.use('seaborn-v0_8-pastel')
    
    # Ensure output folder exists
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # Read CSV file
    print(f"Reading data file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check if necessary columns exist
    if 'file_name' not in df.columns:
        raise ValueError("Data is missing 'file_name' column")
    if channel not in df.columns:
        raise ValueError(f"Data is missing '{channel}' column")

    # Group by file name and calculate channel mean/std/count
    channel_stats = df.groupby('file_name')[channel].agg(['mean', 'std', 'count']).reset_index()
    channel_stats = channel_stats.rename(columns={
        'mean': f'{channel} Mean',
        'std': f'{channel} Std',
        'count': 'Sample Size'
    })

    # Calculate 95% confidence interval (using std)
    channel_stats[f'{channel} CI'] = 1.96 * channel_stats[f'{channel} Std'] / np.sqrt(channel_stats['Sample Size'])

    # Extract well ID if present in file_name
    channel_stats['Well'] = channel_stats['file_name'].str.extract(r'([A-H]\d{1,2})')

    print(f"{channel} group statistics results:")
    print(channel_stats)
    
    # Choose an appealing color
    bar_color = '#4682B4'  # Steel Blue
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    
    # Use seaborn to generate bar chart with single color
    ax = sns.barplot(
        x='file_name', 
        y=f'{channel} Mean', 
        data=channel_stats,
        color=bar_color,
        capsize=0.1,
        alpha=0.8
    )
    
    # Add error bars (95% confidence interval)
    for i, row in channel_stats.iterrows():
        ax.errorbar(
            i, row[f'{channel} Mean'], 
            yerr=row[f'{channel} CI'], 
            fmt='none', 
            c='black', 
            capsize=5
        )
    
    # Adjust x-axis labels if there are many files
    if len(channel_stats) > 10:
        plt.xticks(rotation=45, ha='right')
    
    # Set chart title and labels
    plt.title(f'Average {channel} Values by File', fontsize=14)
    plt.xlabel('File Name', fontsize=12)
    plt.ylabel(f'{channel} Mean Value', fontsize=12)

    # Add data labels
    for i, v in enumerate(channel_stats[f'{channel} Mean']):
        ax.text(i, v + 0.1, f"{v:.2f}", ha='center', fontsize=9)
    
    # Optimize layout
    plt.tight_layout()
    
    # Save chart (if output folder is specified)
    if output_folder:
        output_path = os.path.join(output_folder, f'{channel}_by_file.png')
        plt.savefig(output_path, dpi=300)
        print(f"Bar chart saved to: {output_path}")
    
    # Display chart
    plt.show()
    
    # Create logarithmic scale version (if data range is large)
    if channel_stats[f'{channel} Mean'].max() / channel_stats[f'{channel} Mean'].min() > 10:
        plt.figure(figsize=(12, 6))
        
        # Use seaborn to generate bar chart (log scale)
        ax = sns.barplot(
            x='file_name', 
            y=f'{channel} Mean', 
            data=channel_stats,
            color=bar_color,
            capsize=0.1,
            alpha=0.8
        )
        
        # Set logarithmic scale
        plt.yscale('log')
        
        # Format y-axis labels as normal numbers (not scientific notation)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        
        # Adjust x-axis labels if there are many files
        if len(channel_stats) > 10:
            plt.xticks(rotation=45, ha='right')
        
        # Set chart title and labels
        plt.title(f'Average {channel} Values by File (Log Scale)', fontsize=14)
        plt.xlabel('File Name', fontsize=12)
        plt.ylabel(f'{channel} Mean Value (Log Scale)', fontsize=12)

        # Optimize layout
        plt.tight_layout()
        
        # Save chart (if output folder is specified)
        if output_folder:
            output_path = os.path.join(output_folder, f'{channel}_by_file_log_scale.png')
            plt.savefig(output_path, dpi=300)
            print(f"Log scale bar chart saved to: {output_path}")
        
        # Display chart
        plt.show()
    
    # Create version sorted by sample size
    plt.figure(figsize=(12, 6))
    
    # Sort by sample size
    channel_stats_sorted = channel_stats.sort_values('Sample Size', ascending=False)
    
    # Generate bar chart
    ax = sns.barplot(
        x='file_name', 
        y=f'{channel} Mean', 
        data=channel_stats_sorted,
        order=channel_stats_sorted['file_name'],
        color=bar_color,
        capsize=0.1,
        alpha=0.8
    )
    
    # Add error bars
    for i, (_, row) in enumerate(channel_stats_sorted.iterrows()):
        ax.errorbar(
            i, row[f'{channel} Mean'], 
            yerr=row[f'{channel} CI'], 
            fmt='none', 
            c='black', 
            capsize=5
        )
    
    # Adjust x-axis labels if there are many files
    if len(channel_stats) > 10:
        plt.xticks(rotation=45, ha='right')
    
    # Set chart title and labels
    plt.title(f'Average {channel} Values by File (Sorted by Sample Size)', fontsize=14)
    plt.xlabel('File Name', fontsize=12)
    plt.ylabel(f'{channel} Mean Value', fontsize=12)

    # Optimize layout
    plt.tight_layout()
    
    # Save chart (if output folder is specified)
    if output_folder:
        output_path = os.path.join(output_folder, f'{channel}_by_file_sorted.png')
        plt.savefig(output_path, dpi=300)
        print(f"Bar chart sorted by sample size saved to: {output_path}")
    
    # Display chart
    plt.show()
    
    # Output CSV statistics
    if output_folder:
        stats_path = os.path.join(output_folder, f'{channel}_statistics.csv')
        channel_stats.to_csv(stats_path, index=False)
        print(f"{channel} statistics saved to: {stats_path}")

    return channel_stats


def plot_channel_heatmap(
    csv_path: str | None = None,
    data: pd.DataFrame | None = None,
    output_folder: str | None = None,
    channel: str = 'FL8-A',
    agg: str | Callable = 'mean',
    cmap: str = 'viridis',
    value_log: bool = False,
    value_log_base: float = 10.0,
    annotate: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    robust: bool = False,
    title: str | None = None,
    min_count_threshold: int = 1000,
    plot_counts: bool = True,
    plot_std: bool = True,
    count_cmap: str = 'Blues',
    std_cmap: str = 'mako',
    std_log: bool = True,
    std_log_base: float = 10.0,
):
    """
    Plot a 96-well plate heatmap for a given channel and save to the output folder.

    Assumptions:
    - Each row in the CSV is a single cell/event with columns including `file_name` and the target channel.
    - The well ID can be found in either an existing `Well` column (preferred) or extracted from `file_name`.
      Expected well format like 'A1'..'H12' (case-insensitive).

    Parameters:
    - csv_path: Path to merged processed CSV data.
    - output_folder: If provided, save the heatmap PNG and the aggregated CSV here.
    - channel: Channel column to aggregate and plot (e.g., 'FL8-A').
    - agg: Aggregation for the channel per well: one of {'mean','median','sum','min','max','count'} or a callable.
    - cmap: Matplotlib/Seaborn colormap for heatmap.
    - annotate: Whether to annotate each well with its value.
    - vmin, vmax: Optional fixed color scale limits. If None, Seaborn defaults (with `robust` if True).
    - robust: If True and vmin/vmax are None, use robust quantiles for color scaling (ignore outliers).
    - title: Optional plot title override.

    Returns:
    - pd.DataFrame with columns: ['Well', 'Row', 'Col', 'Value'] for the aggregated values.
    """
    # Style
    plt.style.use('seaborn-v0_8-pastel')

    # Ensure output folder exists
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    if data is None:
        if csv_path is None:
            raise ValueError("Either csv_path or data must be provided")
        print(f"Reading data file: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = data.copy()

    if channel not in df.columns:
        raise ValueError(f"Data is missing '{channel}' column")

    # Determine/construct Well column
    if 'Well' in df.columns:
        well_series = df['Well'].astype(str)
    elif 'file_name' in df.columns:
        # Improved extraction: match A10-A12 before A1-A9 to avoid premature stopping
        well_series = df['file_name'].astype(str).str.extract(r'([A-Ha-h](?:1[0-2]|0?[1-9]))', expand=False)
    else:
        raise ValueError("Data is missing both 'Well' and 'file_name' columns; cannot infer wells.")

    df = df.copy()
    df['Well'] = well_series.str.upper()

    # Filter to valid 96-well wells (A-H, 1-12)
    valid_rows = list('ABCDEFGH')
    valid_cols = list(range(1, 13))
    # Split row/col
    df['Row'] = df['Well'].str[0]
    df['Col'] = pd.to_numeric(df['Well'].str.extract(r'([0-9]{1,2})')[0], errors='coerce')
    df = df[df['Row'].isin(valid_rows) & df['Col'].isin(valid_cols)]

    # If no valid rows after parsing, we will still render a 96-well grid with zeros

    # Aggregate per well
    if not df.empty:
        if isinstance(agg, str):
            if agg not in {'mean', 'median', 'sum', 'min', 'max', 'count'}:
                raise ValueError("agg must be one of {'mean','median','sum','min','max','count'} or a callable")
            agg_series = df.groupby(['Row', 'Col'])[channel].agg(agg)
        elif callable(agg):
            agg_series = df.groupby(['Row', 'Col'])[channel].apply(agg)
        else:
            raise ValueError("Invalid agg parameter. Provide a string or a callable.")
        count_series = df.groupby(['Row', 'Col'])[channel].count()
        std_series = df.groupby(['Row', 'Col'])[channel].std(ddof=1)
        agg_df = agg_series.reset_index().rename(columns={channel: 'Value'})
        count_df = count_series.reset_index().rename(columns={channel: 'Count'})
        std_df = std_series.reset_index().rename(columns={channel: 'Std'})
        agg_df = agg_df.merge(count_df, on=['Row', 'Col'], how='left')
        agg_df = agg_df.merge(std_df, on=['Row', 'Col'], how='left')
    else:
        # Build empty frames with required columns
        agg_df = pd.DataFrame(columns=['Row', 'Col', 'Value', 'Count', 'Std'])

    # Create an 8x12 grid DataFrame in plate order (A-H rows, 1-12 columns)
    grid_index = pd.Index(valid_rows, name='Row')
    grid_columns = pd.Index(valid_cols, name='Col')
    grid_value = pd.DataFrame(index=grid_index, columns=grid_columns, dtype=float)
    grid_count = pd.DataFrame(index=grid_index, columns=grid_columns, dtype=float)
    grid_std = pd.DataFrame(index=grid_index, columns=grid_columns, dtype=float)
    for _, r in agg_df.iterrows():
        row, col = r['Row'], int(r['Col'])
        grid_value.at[row, col] = r['Value']
        grid_count.at[row, col] = r.get('Count', np.nan)
        grid_std.at[row, col] = r.get('Std', np.nan)

    # Fill missing wells with 0 for value and 0 for count (show full 12 columns)
    grid_value = grid_value.fillna(0.0)
    grid_count = grid_count.fillna(0.0)
    grid_std = grid_std.fillna(0.0)

    # Optional log transform of aggregated values (after fillna with 0 -> add epsilon)
    if value_log:
        epsilon_val = 1e-9
        if value_log_base == 10.0:
            grid_value_plot = np.log10(grid_value + epsilon_val)
        else:
            grid_value_plot = np.log(grid_value + epsilon_val) / np.log(value_log_base)
    else:
        grid_value_plot = grid_value

    # Optional log transform of std values (add small epsilon to avoid log(0))
    if plot_std and std_log:
        epsilon = 1e-9
        if std_log_base == 10.0:
            grid_std_log = np.log10(grid_std + epsilon)
        else:
            grid_std_log = np.log(grid_std + epsilon) / np.log(std_log_base)
    else:
        grid_std_log = grid_std.copy()

    # Build heatmap
    plt.figure(figsize=(12, 7.5))  # 12 columns x 8 rows; keep cells roughly square
    ax = sns.heatmap(
        grid_value_plot,
        cmap=cmap,
        annot=False,  # we'll handle annotations manually to overlay count flags
        fmt='',
        linewidths=0.5,
        linecolor='white',
        cbar=True,
        square=True,
        vmin=vmin,
        vmax=vmax,
        robust=(robust and vmin is None and vmax is None),
        # no mask since we fill with zeros to always show the full plate
    )

    # Aesthetic tweaks: label ticks as plate coordinates
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    # Ensure all ticks are shown explicitly
    ax.set_xticks(np.arange(len(grid_columns)) + 0.5)
    ax.set_xticklabels([str(c) for c in grid_columns], rotation=0)
    ax.set_yticks(np.arange(len(grid_index)) + 0.5)
    ax.set_yticklabels(list(grid_index), rotation=0)

    # Add value annotations and low-count flags
    if annotate:
        for i, r in enumerate(grid_index):
            for j, c in enumerate(grid_columns):
                val = grid_value_plot.at[r, c]
                # Value in black only (do not mix with count flags)
                ax.text(j + 0.5, i + 0.5, f"{val:.2f}", color='black', ha='center', va='center', fontsize=8)

    # Title
    if title is None:
        base_t = f"{channel} {agg if isinstance(agg, str) else 'agg'} per well"
        if value_log:
            base_t += f" (log base {value_log_base:g})"
        title = base_t + " (96‑well plate)"
    plt.title(title, fontsize=14)
    plt.tight_layout()

    # Save plot
    if output_folder:
        heatmap_fn = f"{channel}_plate_heatmap_{agg if isinstance(agg, str) else 'agg'}{'_log' if value_log else ''}.png"
        heatmap_path = os.path.join(output_folder, heatmap_fn)
        plt.savefig(heatmap_path, dpi=300)
        print(f"Heatmap saved to: {heatmap_path}")

    # Show plot
    plt.show()

    # Save aggregated CSV
    out_value = (
        grid_value.reset_index()
        .melt(id_vars='Row', var_name='Col', value_name='Value')
    )
    out_count = (
        grid_count.reset_index()
        .melt(id_vars='Row', var_name='Col', value_name='Count')
    )
    out_std = (
        grid_std.reset_index()
        .melt(id_vars='Row', var_name='Col', value_name='Std')
    )
    out_std_log = (
        grid_std_log.reset_index()
        .melt(id_vars='Row', var_name='Col', value_name='StdLog')
    )
    out_records = (
        out_value.merge(out_count, on=['Row', 'Col'], how='left')
        .merge(out_std, on=['Row', 'Col'], how='left')
        .merge(out_std_log, on=['Row', 'Col'], how='left')
        .assign(Col=lambda d: d['Col'].astype(int))
    )
    out_records['Well'] = out_records['Row'] + out_records['Col'].astype(int).astype(str)
    out_records['LowCountFlag'] = out_records['Count'] < float(min_count_threshold)
    if value_log:
        out_records['ValueLog'] = out_records['Value'].apply(lambda x: (np.log10(x + 1e-9) if value_log_base == 10.0 else np.log(x + 1e-9) / np.log(value_log_base)))
    cols_order = ['Well','Row','Col','Value'] + (['ValueLog'] if value_log else []) + ['Count','Std','StdLog','LowCountFlag']
    out_records = out_records[cols_order].sort_values(['Row', 'Col']).reset_index(drop=True)

    # Plot counts heatmap separately
    if plot_counts:
        plt.figure(figsize=(12, 7.5))
        ax_cnt = sns.heatmap(
            grid_count,
            cmap=count_cmap,
            annot=False,
            fmt='',
            linewidths=0.5,
            linecolor='white',
            cbar=True,
            square=True,
        )
        ax_cnt.set_xlabel('Column', fontsize=12)
        ax_cnt.set_ylabel('Row', fontsize=12)
        ax_cnt.set_xticks(np.arange(len(grid_columns)) + 0.5)
        ax_cnt.set_xticklabels([str(c) for c in grid_columns], rotation=0)
        ax_cnt.set_yticks(np.arange(len(grid_index)) + 0.5)
        ax_cnt.set_yticklabels(list(grid_index), rotation=0)
        # Annotate counts; low counts in red
        for i, r in enumerate(grid_index):
            for j, c in enumerate(grid_columns):
                cnt = grid_count.at[r, c]
                color = 'red' if cnt < float(min_count_threshold) else 'black'
                ax_cnt.text(j + 0.5, i + 0.5, f"{int(cnt):d}", color=color, ha='center', va='center', fontsize=8)
        plt.title(f"Sample size per well (threshold {min_count_threshold})", fontsize=14)
        plt.tight_layout()
        if output_folder:
            cnt_path = os.path.join(output_folder, f"{channel}_plate_heatmap_counts.png")
            plt.savefig(cnt_path, dpi=300)
            print(f"Counts heatmap saved to: {cnt_path}")
        plt.show()

    # Plot std heatmap separately
    if plot_std:
        plt.figure(figsize=(12, 7.5))
        ax_std = sns.heatmap(
            grid_std_log if std_log else grid_std,
            cmap=std_cmap,
            annot=False,
            fmt='',
            linewidths=0.5,
            linecolor='white',
            cbar=True,
            square=True,
        )
        ax_std.set_xlabel('Column', fontsize=12)
        ax_std.set_ylabel('Row', fontsize=12)
        ax_std.set_xticks(np.arange(len(grid_columns)) + 0.5)
        ax_std.set_xticklabels([str(c) for c in grid_columns], rotation=0)
        ax_std.set_yticks(np.arange(len(grid_index)) + 0.5)
        ax_std.set_yticklabels(list(grid_index), rotation=0)
        if annotate:
            for i, r in enumerate(grid_index):
                for j, c in enumerate(grid_columns):
                    stdv = (grid_std_log.at[r, c] if std_log else grid_std.at[r, c])
                    ax_std.text(j + 0.5, i + 0.5, f"{stdv:.2f}", color='black', ha='center', va='center', fontsize=8)
        plt.title(f"{'Log ' if std_log else ''}Standard deviation of {channel} per well", fontsize=14)
        plt.tight_layout()
        if output_folder:
            std_path = os.path.join(output_folder, f"{channel}_plate_heatmap_std.png")
            plt.savefig(std_path, dpi=300)
            print(f"Std heatmap saved to: {std_path}")
        plt.show()

    if output_folder:
        csv_out = os.path.join(output_folder, f"{channel}_plate_aggregated_{agg if isinstance(agg, str) else 'agg'}.csv")
        out_records.to_csv(csv_out, index=False)
        print(f"Aggregated well values saved to: {csv_out}")

    return out_records


def plot_channel_heatmaps_by_plate(
    csv_path: str,
    output_folder: str | None = None,
    channel: str = 'FL8-A',
    agg: str | Callable = 'mean',
    plate_regex: str = r'^(?P<plate>\d{2})-Well-(?P<well>[A-Ha-h](?:1[0-2]|0?[1-9]))',
    **heatmap_kwargs,
) -> Dict[str, pd.DataFrame]:
    """Generate per-plate 96-well heatmaps and aggregated CSVs.

    This wrapper splits the input merged events table by plate identifier extracted from
    the `file_name` (e.g., "01-Well-A1.fcs" -> plate "01", well "A1") and calls
    `plot_channel_heatmap` for each plate subset, producing separate output folders.

    Parameters
    ----------
    csv_path : str
        Path to merged processed CSV containing per-event data with a `file_name` column.
    output_folder : str | None
        Root output folder. If provided, per-plate subfolders will be created inside it.
    channel : str
        Channel name to aggregate (column must exist in data).
    agg : str | Callable
        Aggregation spec passed through to `plot_channel_heatmap`.
    plate_regex : str
        Regex with named groups 'plate' and 'well' to extract identifiers from `file_name`.
    **heatmap_kwargs : dict
        Additional keyword arguments forwarded to `plot_channel_heatmap` (e.g., cmap, annotate...).

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping of plate_id -> aggregated per-well DataFrame returned by `plot_channel_heatmap`.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV path not found: {csv_path}")
    print(f"Reading data file for multi-plate heatmaps: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'file_name' not in df.columns:
        raise ValueError("Data must contain 'file_name' column for plate/well extraction")
    if channel not in df.columns:
        raise ValueError(f"Data is missing '{channel}' column")

    pattern = re.compile(plate_regex)
    extracted = df['file_name'].str.extract(pattern)
    if extracted.empty or 'plate' not in extracted.columns or 'well' not in extracted.columns:
        raise ValueError("Plate regex did not extract 'plate' and 'well' groups from file_name")
    df['Plate'] = extracted['plate']
    df['Well'] = extracted['well'].str.upper()

    plate_ids = sorted(df['Plate'].dropna().unique())
    if not plate_ids:
        # Fallback: treat all data as one plate labeled 'ALL'
        plate_ids = ['ALL']
        df['Plate'] = 'ALL'

    results: Dict[str, pd.DataFrame] = {}
    for plate_id in plate_ids:
        sub = df[df['Plate'] == plate_id].copy()
        if output_folder:
            plate_out = os.path.join(output_folder, f"plate_{plate_id}")
        else:
            plate_out = None
        print(f"\nGenerating heatmaps for plate {plate_id} with {len(sub)} events...")
        res = plot_channel_heatmap(
            csv_path=None,
            data=sub,
            output_folder=plate_out,
            channel=channel,
            agg=agg,
            **heatmap_kwargs,
        )
        results[plate_id] = res
    return results


def plot_channel_heatmap_min_count(
    csv_path: str | None = None,
    data: pd.DataFrame | None = None,
    output_folder: str | None = None,
    channel: str = 'FL8-A',
    threshold: int = 1000,
    agg: str | Callable = 'mean',
    cmap: str = 'viridis',
    value_log: bool = False,
    value_log_base: float = 10.0,
    annotate: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    robust: bool = False,
    title: str | None = None,
    plot_counts: bool = True,
    plot_std: bool = True,
    count_cmap: str = 'Blues',
    std_cmap: str = 'mako',
    std_log: bool = True,
    std_log_base: float = 10.0,
):
    """
    Plot a 96-well heatmap where wells with sample size < threshold are zeroed.

    This function first computes per-well sample counts, keeps only wells with
    count >= threshold, and drops all events from low-count wells. Missing wells
    are rendered as zeros by the underlying heatmap function.

    Parameters
    ----------
    csv_path / data : provide one of them as input events table (with `file_name` or `Well`).
    threshold : int
        Minimum per-well event count to display actual values; below this value, wells will show 0.
    Other parameters mirror plot_channel_heatmap.
    """
    # Load data
    if data is None:
        if csv_path is None:
            raise ValueError("Either csv_path or data must be provided")
        print(f"Reading data file for min-count filtering: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = data.copy()

    if channel not in df.columns:
        raise ValueError(f"Data is missing '{channel}' column")

    # Determine Well like in plot_channel_heatmap
    if 'Well' in df.columns:
        well_series = df['Well'].astype(str)
    elif 'file_name' in df.columns:
        well_series = df['file_name'].astype(str).str.extract(r'([A-Ha-h](?:1[0-2]|0?[1-9]))', expand=False)
    else:
        raise ValueError("Data is missing both 'Well' and 'file_name' columns; cannot infer wells.")
    df['Well'] = well_series.str.upper()
    df['Row'] = df['Well'].str[0]
    df['Col'] = pd.to_numeric(df['Well'].str.extract(r'([0-9]{1,2})')[0], errors='coerce')

    # Compute per-well counts and filter
    valid_rows = list('ABCDEFGH')
    valid_cols = list(range(1, 13))
    df = df[df['Row'].isin(valid_rows) & df['Col'].isin(valid_cols)]
    if df.empty:
        print("No valid wells found; plotting empty plate with zeros.")
        filtered = df
    else:
        counts = df.groupby(['Row', 'Col'])[channel].count().reset_index(name='Count')
        keep = counts[counts['Count'] >= int(threshold)][['Row', 'Col']]
        if keep.empty:
            print(f"No wells meet threshold >= {threshold}; plotting all zeros.")
            filtered = df.iloc[0:0].copy()  # empty to force zeros
        else:
            # Keep only events from wells meeting count threshold
            key = keep.assign(key=1).merge(
                df[['Row', 'Col', channel]], on=['Row', 'Col'], how='left'
            )[['Row', 'Col']].drop_duplicates()
            filtered = df.merge(key, on=['Row', 'Col'], how='inner')

    # Prepare output folder to avoid overwriting other runs
    out_dir = os.path.join(output_folder, f"minCount_{threshold}") if output_folder else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Compose title
    final_title = title or f"{channel} {agg if isinstance(agg, str) else 'agg'} per well (count >= {threshold})"

    # Delegate to base heatmap (it will fill missing wells as zeros)
    return plot_channel_heatmap(
        csv_path=None,
        data=filtered,
        output_folder=out_dir,
        channel=channel,
        agg=agg,
        cmap=cmap,
        value_log=value_log,
        value_log_base=value_log_base,
        annotate=annotate,
        vmin=vmin,
        vmax=vmax,
        robust=robust,
        title=final_title,
        min_count_threshold=threshold,
        plot_counts=plot_counts,
        plot_std=plot_std,
        count_cmap=count_cmap,
        std_cmap=std_cmap,
        std_log=std_log,
        std_log_base=std_log_base,
    )


def plot_channel_heatmaps_by_plate_min_count(
    csv_path: str,
    output_folder: str | None = None,
    channel: str = 'FL8-A',
    threshold: int = 1000,
    agg: str | Callable = 'mean',
    plate_regex: str = r'^(?P<plate>\d{2})-Well-(?P<well>[A-Ha-h](?:1[0-2]|0?[1-9]))',
    **heatmap_kwargs,
):
    """Per-plate heatmaps with min-count filtering.

    For each plate (e.g., 01-, 02- prefix in file_name), keep wells with sample
    count >= threshold and set other wells to 0 by omission. Generates separate
    output under output_folder/plate_{plate}/minCount_{threshold}.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV path not found: {csv_path}")
    print(f"Reading data file for multi-plate min-count heatmaps: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'file_name' not in df.columns:
        raise ValueError("Data must contain 'file_name' column for plate/well extraction")
    if channel not in df.columns:
        raise ValueError(f"Data is missing '{channel}' column")

    pattern = re.compile(plate_regex)
    extracted = df['file_name'].str.extract(pattern)
    if extracted.empty or 'plate' not in extracted.columns or 'well' not in extracted.columns:
        raise ValueError("Plate regex did not extract 'plate' and 'well' groups from file_name")
    df['Plate'] = extracted['plate']
    df['Well'] = extracted['well'].str.upper()

    plate_ids = sorted(df['Plate'].dropna().unique())
    if not plate_ids:
        plate_ids = ['ALL']
        df['Plate'] = 'ALL'

    results: Dict[str, pd.DataFrame] = {}
    for plate_id in plate_ids:
        sub = df[df['Plate'] == plate_id].copy()
        plate_out = os.path.join(output_folder, f"plate_{plate_id}") if output_folder else None
        print(f"\nGenerating min-count heatmaps for plate {plate_id} with threshold {threshold} and {len(sub)} events...")
        res = plot_channel_heatmap_min_count(
            csv_path=None,
            data=sub,
            output_folder=plate_out,
            channel=channel,
            threshold=threshold,
            agg=agg,
            **heatmap_kwargs,
        )
        results[plate_id] = res
    return results


def plot_plate_ratio_heatmap(
    output_root: str,
    channel: str = 'FL8-A',
    agg: str | Callable = 'mean',
    plate_a: str = '01',
    plate_b: str = '02',
    threshold: int = 1000,
    cmap: str = 'coolwarm',
    annotate: bool = True,
    value_log: bool = False,
    value_log_base: float = 10.0,
    vmin: float | None = None,
    vmax: float | None = None,
    robust: bool = False,
    title: str | None = None,
):
    """
    Compute per-well ratio PlateA/PlateB from minCount-threshold aggregated CSVs and plot a 96-well heatmap.

    This function expects prior runs of `plot_channel_heatmaps_by_plate_min_count` or `plot_channel_heatmap_min_count`
    that produced per-plate aggregated CSVs under:
        {output_root}/plate_{plate}/minCount_{threshold}/{channel}_plate_aggregated_{agg}.csv

    Parameters
    ----------
    output_root : str
        Root folder containing per-plate outputs (e.g., D:\\FACS_analysis\\<exp>\\analysis_results).
    channel : str
        Channel to use for ratio (must match the aggregated CSV filenames).
    agg : str | Callable
        Aggregation used when generating those CSVs. If callable, will look for suffix 'agg'.
    plate_a, plate_b : str
        Plate identifiers (e.g., '01', '02'). Ratio computed as A/B.
    threshold : int
        The minCount subfolder to read from (e.g., 1000 -> minCount_1000).
    cmap : str
        Colormap for the ratio heatmap.
    annotate : bool
        Overlay numeric values on the heatmap.
    value_log : bool
        Optional log transform for ratio values (default False).
    value_log_base : float
        Base for the logarithm when value_log is True.
    vmin, vmax, robust, title :
        Passed to seaborn.heatmap / used for figure title.

    Returns
    -------
    pd.DataFrame
        DataFrame with per-well values for PlateA, PlateB, Ratio (and RatioLog if requested).
    """
    # Resolve expected CSV paths
    agg_tag = agg if isinstance(agg, str) else 'agg'
    a_csv = os.path.join(output_root, f"plate_{plate_a}", f"minCount_{threshold}", f"{channel}_plate_aggregated_{agg_tag}.csv")
    b_csv = os.path.join(output_root, f"plate_{plate_b}", f"minCount_{threshold}", f"{channel}_plate_aggregated_{agg_tag}.csv")
    if not os.path.isfile(a_csv):
        raise FileNotFoundError(f"Missing CSV for plate {plate_a}: {a_csv}")
    if not os.path.isfile(b_csv):
        raise FileNotFoundError(f"Missing CSV for plate {plate_b}: {b_csv}")

    print(f"Reading aggregated CSVs for ratio: \n - A: {a_csv}\n - B: {b_csv}")
    df_a = pd.read_csv(a_csv)
    df_b = pd.read_csv(b_csv)

    # Ensure required columns
    for name, df in [('A', df_a), ('B', df_b)]:
        for col in ['Row', 'Col', 'Value']:
            if col not in df.columns:
                raise ValueError(f"CSV for plate {name} is missing required column '{col}'")

    # Prepare full plate indices
    rows = list('ABCDEFGH')
    cols = list(range(1, 13))

    # Merge per-well values; fill missing as 0
    a_sel = df_a[['Row', 'Col', 'Value']].rename(columns={'Value': 'ValueA'})
    b_sel = df_b[['Row', 'Col', 'Value']].rename(columns={'Value': 'ValueB'})
    merged = (
        pd.DataFrame([(r, c) for r in rows for c in cols], columns=['Row', 'Col'])
        .merge(a_sel, on=['Row', 'Col'], how='left')
        .merge(b_sel, on=['Row', 'Col'], how='left')
        .fillna({'ValueA': 0.0, 'ValueB': 0.0})
    )

    # Ratio with zero-handling: if numerator<=0 or denominator<=0 -> 0
    val_a = merged['ValueA'].astype(float)
    val_b = merged['ValueB'].astype(float)
    ratio = np.where((val_a <= 0) | (val_b <= 0), 0.0, val_a / val_b)
    merged['Ratio'] = ratio

    if value_log:
        eps = 1e-9
        if value_log_base == 10.0:
            merged['RatioLog'] = np.log10(merged['Ratio'] + eps)
        else:
            merged['RatioLog'] = np.log(merged['Ratio'] + eps) / np.log(value_log_base)

    # Build grid for plotting
    grid_index = pd.Index(rows, name='Row')
    grid_columns = pd.Index(cols, name='Col')
    grid_ratio = pd.DataFrame(index=grid_index, columns=grid_columns, dtype=float)
    for _, r in merged.iterrows():
        grid_ratio.at[r['Row'], int(r['Col'])] = r['Ratio']
    grid_ratio = grid_ratio.fillna(0.0)

    grid_plot = grid_ratio
    if value_log:
        grid_plot = merged.pivot(index='Row', columns='Col', values='RatioLog').reindex(index=rows, columns=cols).fillna(0.0)

    # Plot heatmap
    plt.style.use('seaborn-v0_8-pastel')
    plt.figure(figsize=(12, 7.5))
    ax = sns.heatmap(
        grid_plot,
        cmap=cmap,
        annot=False,
        fmt='',
        linewidths=0.5,
        linecolor='white',
        cbar=True,
        square=True,
        vmin=vmin,
        vmax=vmax,
        robust=(robust and vmin is None and vmax is None),
    )
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_xticks(np.arange(len(grid_columns)) + 0.5)
    ax.set_xticklabels([str(c) for c in grid_columns], rotation=0)
    ax.set_yticks(np.arange(len(grid_index)) + 0.5)
    ax.set_yticklabels(list(grid_index), rotation=0)

    if annotate:
        for i, r in enumerate(grid_index):
            for j, c in enumerate(grid_columns):
                val = grid_plot.at[r, c]
                ax.text(j + 0.5, i + 0.5, f"{val:.2f}", color='black', ha='center', va='center', fontsize=8)

    # Title
    if title is None:
        base = f"{channel} ratio plate {plate_a}/{plate_b} (minCount {threshold})"
        if value_log:
            base += f" (log base {value_log_base:g})"
        title = base
    plt.title(title, fontsize=14)
    plt.tight_layout()

    # Output dir for ratio artifacts
    ratio_dir = os.path.join(output_root, f"ratio_{plate_a}_over_{plate_b}", f"minCount_{threshold}")
    os.makedirs(ratio_dir, exist_ok=True)

    # Save plot
    ratio_png = os.path.join(ratio_dir, f"{channel}_ratio_{plate_a}_over_{plate_b}{'_log' if value_log else ''}.png")
    plt.savefig(ratio_png, dpi=300)
    print(f"Ratio heatmap saved to: {ratio_png}")
    plt.show()

    # Save ratio CSV
    out = merged.copy()
    out['Well'] = out['Row'] + out['Col'].astype(int).astype(str)
    cols = ['Well', 'Row', 'Col', 'ValueA', 'ValueB', 'Ratio'] + (['RatioLog'] if value_log else [])
    out = out[cols].sort_values(['Row', 'Col']).reset_index(drop=True)
    ratio_csv = os.path.join(ratio_dir, f"{channel}_ratio_{plate_a}_over_{plate_b}.csv")
    out.to_csv(ratio_csv, index=False)
    print(f"Ratio CSV saved to: {ratio_csv}")

    return out


def plot_plate_ratio_bars(
    output_root: str,
    channel: str = 'FL8-A',
    agg: str | Callable = 'mean',
    plate_a: str = '01',
    plate_b: str = '02',
    threshold: int = 1000,
    nonzero_only: bool = True,
    colors: tuple = ('#1f77b4', '#ff7f0e'),
    figsize_values: tuple = (16, 6),
    figsize_ratio: tuple = (16, 6),
    rotate_xticks: int = 90,
    sort: str | None = None,  # None | 'well' | 'ratio-asc' | 'ratio-desc'
    annotate: bool = False,
    values_log: bool = False,
    ratio_log: bool = False,
):
    """
    Plot two bar charts using previously aggregated per-plate values:
      1) Side-by-side bars for plate A (01) and plate B (02) per Well.
      2) Fold change (ratio = A/B) per Well.

    Only wells with non-zero ratio are plotted when `nonzero_only=True`.

    Data is loaded from the same aggregated CSVs as `plot_plate_ratio_heatmap`:
      {output_root}/plate_{plate}/minCount_{threshold}/{channel}_plate_aggregated_{agg}.csv

    Parameters
    ----------
    output_root : str
        Root folder containing per-plate outputs.
    channel : str
        Channel used when aggregating per-well values.
    agg : str | Callable
        Aggregation used earlier (filename suffix). If callable, uses 'agg'.
    plate_a, plate_b : str
        Plate identifiers (e.g., '01', '02').
    threshold : int
        minCount subfolder to read from.
    nonzero_only : bool
        If True, keep only wells with Ratio > 0.
    colors : tuple
        Colors for plate A and plate B bars.
    figsize_values / figsize_ratio : tuple
        Figure sizes for the two plots.
    rotate_xticks : int
        X-axis label rotation angle.
    sort : str | None
        Optional sort: 'well' keeps plate order A1..H12; 'ratio-asc'/'ratio-desc' sorts by ratio.
    annotate : bool
        Whether to annotate bar heights.

        Saves two PNGs under:
      {output_root}/ratio_{plate_a}_over_{plate_b}/minCount_{threshold}/
        - {channel}_bars_{plate_a}_vs_{plate_b}.png
        - {channel}_fold_change_{plate_a}_over_{plate_b}.png
    Returns
    -------
    pd.DataFrame
        DataFrame with Well, ValueA, ValueB, Ratio used for plotting.
    """
    plt.style.use('seaborn-v0_8-pastel')

    agg_tag = agg if isinstance(agg, str) else 'agg'
    a_csv = os.path.join(output_root, f"plate_{plate_a}", f"minCount_{threshold}", f"{channel}_plate_aggregated_{agg_tag}.csv")
    b_csv = os.path.join(output_root, f"plate_{plate_b}", f"minCount_{threshold}", f"{channel}_plate_aggregated_{agg_tag}.csv")
    if not os.path.isfile(a_csv):
        raise FileNotFoundError(f"Missing CSV for plate {plate_a}: {a_csv}")
    if not os.path.isfile(b_csv):
        raise FileNotFoundError(f"Missing CSV for plate {plate_b}: {b_csv}")

    df_a = pd.read_csv(a_csv)
    df_b = pd.read_csv(b_csv)
    for name, df in [('A', df_a), ('B', df_b)]:
        for col in ['Row', 'Col', 'Value']:
            if col not in df.columns:
                raise ValueError(f"CSV for plate {name} is missing required column '{col}'")

    # Build full well order
    rows = list('ABCDEFGH')
    cols = list(range(1, 13))
    well_order = [f"{r}{c}" for r in rows for c in cols]

    a_sel = df_a[['Row', 'Col', 'Value']].rename(columns={'Value': 'ValueA'})
    b_sel = df_b[['Row', 'Col', 'Value']].rename(columns={'Value': 'ValueB'})
    merged = (
        pd.DataFrame([(r, c) for r in rows for c in cols], columns=['Row', 'Col'])
        .merge(a_sel, on=['Row', 'Col'], how='left')
        .merge(b_sel, on=['Row', 'Col'], how='left')
        .fillna({'ValueA': 0.0, 'ValueB': 0.0})
    )
    merged['Well'] = merged['Row'] + merged['Col'].astype(int).astype(str)
    val_a = merged['ValueA'].astype(float)
    val_b = merged['ValueB'].astype(float)
    merged['Ratio'] = np.where((val_a <= 0) | (val_b <= 0), 0.0, val_a / val_b)

    if nonzero_only:
        data_used = merged[merged['Ratio'] > 0].copy()
    else:
        data_used = merged.copy()

    # Sorting
    if sort == 'ratio-asc':
        data_used = data_used.sort_values('Ratio', ascending=True)
    elif sort == 'ratio-desc':
        data_used = data_used.sort_values('Ratio', ascending=False)
    else:  # keep plate well order
        data_used['__order__'] = data_used['Well'].map({w: i for i, w in enumerate(well_order)})
        data_used = data_used.sort_values('__order__').drop(columns='__order__')

    # Output folder
    ratio_dir = os.path.join(output_root, f"ratio_{plate_a}_over_{plate_b}", f"minCount_{threshold}")
    os.makedirs(ratio_dir, exist_ok=True)

    # 1) Side-by-side values bar plot
    long_vals = pd.melt(
        data_used,
        id_vars=['Well'],
        value_vars=['ValueA', 'ValueB'],
        var_name='Plate',
        value_name='Value'
    )
    long_vals['Plate'] = long_vals['Plate'].map({'ValueA': plate_a, 'ValueB': plate_b})

    plt.figure(figsize=figsize_values)
    ax1 = sns.barplot(
        data=long_vals,
        x='Well', y='Value', hue='Plate',
        palette=[colors[0], colors[1]],
        dodge=True
    )
    ax1.set_title(f"{channel} per-well values: plate {plate_a} vs {plate_b} (minCount {threshold})")
    ax1.set_xlabel('Well')
    ax1.set_ylabel(f'{channel} {agg if isinstance(agg, str) else "agg"}')
    if values_log:
        # With nonzero_only=True, ValueA/ValueB should be > 0; otherwise log scale can't display zeros
        ax1.set_yscale('log')
    if rotate_xticks:
        plt.setp(ax1.get_xticklabels(), rotation=rotate_xticks, ha='right')
    plt.tight_layout()
    values_png = os.path.join(ratio_dir, f"{channel}_bars_{plate_a}_vs_{plate_b}.png")
    plt.savefig(values_png, dpi=300)
    print(f"Values side-by-side bar plot saved to: {values_png}")
    plt.show()

    # 2) Fold change (ratio) bar plot
    plt.figure(figsize=figsize_ratio)
    ax2 = sns.barplot(
        data=data_used,
        x='Well', y='Ratio', color='#6a3d9a'
    )
    ax2.set_title(f"Fold change (plate {plate_a}/{plate_b}) per well (minCount {threshold})")
    ax2.set_xlabel('Well')
    ax2.set_ylabel('Fold change (A/B)')
    if ratio_log:
        ax2.set_yscale('log')
    if rotate_xticks:
        plt.setp(ax2.get_xticklabels(), rotation=rotate_xticks, ha='right')

    if annotate:
        for i, row in enumerate(data_used.itertuples(index=False)):
            ax2.text(i, getattr(row, 'Ratio') + 0.01, f"{getattr(row, 'Ratio'):.2f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    ratio_png = os.path.join(ratio_dir, f"{channel}_fold_change_{plate_a}_over_{plate_b}.png")
    plt.savefig(ratio_png, dpi=300)
    print(f"Fold change bar plot saved to: {ratio_png}")
    plt.show()

    # Save the data used
    out_csv = os.path.join(ratio_dir, f"{channel}_ratio_table_{plate_a}_over_{plate_b}.csv")
    data_used[['Well', 'Row', 'Col', 'ValueA', 'ValueB', 'Ratio']].to_csv(out_csv, index=False)
    print(f"Ratio table saved to: {out_csv}")

    return data_used[['Well', 'Row', 'Col', 'ValueA', 'ValueB', 'Ratio']].reset_index(drop=True)
