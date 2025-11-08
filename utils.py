import os
import pandas as pd
import FlowCal
from glob import glob
import numpy as np
from tqdm import tqdm

def auto_process(df, gating_strategy):
    """
    Apply a series of gating strategies to the dataframe
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    gating_strategy (list): List of gating strategies to apply
    
    Returns:
    pd.DataFrame: Dataframe after processing with all gating strategies
    """
    # Initial filtering - remove non-positive values
    filtered_df = df[(df['FSC-A'] > 0) & (df['SSC-A'] > 0)]
    results = filtered_df.copy()
    
    # Apply sequence of gating strategies
    for gate in gating_strategy:
        results = gate.apply_gate(results)
    
    return results

def process_all_fcs_files(folder_path, gating_strategy, file_pattern='*.fcs'):
    """
    Process all .fcs files in a folder matching the pattern
    
    Parameters:
    folder_path (str): Path to the folder containing .fcs files
    gating_strategy (list): List of gating strategies to apply
    file_pattern (str): File matching pattern, default is '*.fcs'
    
    Returns:
    tuple: (Merged DataFrame, dictionary of data for each file)
    """
    # Get all matching file paths
    file_paths = glob(os.path.join(folder_path, file_pattern))
    
    if not file_paths:
        raise ValueError(f"No files matching '{file_pattern}' found in '{folder_path}'")
    
    print(f"Found {len(file_paths)} files to process")
    
    # Store processed DataFrames for each file
    processed_dfs = {}
    all_dfs = []
    
    # Process each file
    for file_path in tqdm(file_paths, desc="Processing files"):
        file_name = os.path.basename(file_path)
        
        try:
            # Read .fcs file
            data = FlowCal.io.FCSData(file_path)
            
            # Extract metadata (optional)
            metadata = {
                'file_name': file_name,
                'event_count': data.shape[0],
                # Can add more metadata
            }
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=data.channels)
            
            # Add file identifier column
            df['file_name'] = file_name
            
            # Apply gating strategy
            processed_df = auto_process(df, gating_strategy)
            
            # Store results
            processed_dfs[file_name] = {
                'data': processed_df,
                'metadata': metadata,
                'original_event_count': df.shape[0],
                'processed_event_count': processed_df.shape[0]
            }
            
            # Add to merge list
            all_dfs.append(processed_df)
            
            print(f"Processing complete: {file_name} - Original events: {df.shape[0]}, Processed events: {processed_df.shape[0]}")
            
        except Exception as e:
            print(f"Error processing file '{file_name}': {str(e)}")
    
    # Merge all processed DataFrames
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"All files processed. Merged DataFrame contains {combined_df.shape[0]} rows and {combined_df.shape[1]} columns.")
    else:
        combined_df = pd.DataFrame()
        print("No files were successfully processed.")
    
    return combined_df, processed_dfs

def save_results(combined_df, processed_dfs, output_folder, base_filename="processed_data"):
    """
    Save processing results
    
    Parameters:
    combined_df (pd.DataFrame): Merged dataframe
    processed_dfs (dict): Dictionary of processing results for each file
    output_folder (str): Output folder path
    base_filename (str): Base filename
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Save merged DataFrame
    if not combined_df.empty:
        combined_path = os.path.join(output_folder, f"{base_filename}_combined.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"Merged data saved to: {combined_path}")
    
    # Save processing statistics
    stats = []
    for file_name, info in processed_dfs.items():
        stats.append({
            'file_name': file_name,
            'original_events': info['original_event_count'],
            'processed_events': info['processed_event_count'],
            'retention_rate': info['processed_event_count'] / info['original_event_count'] if info['original_event_count'] > 0 else 0
        })
    
    if stats:
        stats_df = pd.DataFrame(stats)
        stats_path = os.path.join(output_folder, f"{base_filename}_statistics.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"Processing statistics saved to: {stats_path}")
    
    # Optional: Save each processed file separately
    for file_name, info in processed_dfs.items():
        file_base = os.path.splitext(file_name)[0]
        individual_path = os.path.join(output_folder, f"{file_base}_processed.csv")
        info['data'].to_csv(individual_path, index=False)

        
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter

def analyze_fl8_by_file(csv_path, output_folder=None):
    """
    Read processed CSV data, calculate FL8-A mean values grouped by file name and generate bar charts
    
    Parameters:
    csv_path (str): Path to the CSV file containing merged processed data
    output_folder (str, optional): Output folder for charts, if not specified, charts will only be displayed
    
    Returns:
    pd.DataFrame: DataFrame containing FL8-A mean, standard deviation, and sample size for each file
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
    if 'FL8-A' not in df.columns:
        raise ValueError("Data is missing 'FL8-A' column")
    
    print(f"Successfully read data with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Group by file name and calculate FL8-A mean
    fl8_stats = df.groupby('file_name')['FL8-A'].agg(['mean', 'std', 'count']).reset_index()
    fl8_stats = fl8_stats.rename(columns={
        'mean': 'FL8-A Mean', 
        'std': 'FL8-A Std', 
        'count': 'Sample Size'
    })
    
    # Calculate 95% confidence interval
    fl8_stats['FL8-A CI'] = 1.96 * fl8_stats['FL8-A Std'] / np.sqrt(fl8_stats['Sample Size'])
    
    # Extract key information from file names (assuming format: "Well-A1.fcs")
    fl8_stats['Well'] = fl8_stats['file_name'].str.extract(r'([A-Z]\d+)')
    
    print("FL8-A group statistics results:")
    print(fl8_stats)
    
    # Choose an appealing color
    bar_color = '#4682B4'  # Steel Blue
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    
    # Use seaborn to generate bar chart with single color
    ax = sns.barplot(
        x='file_name', 
        y='FL8-A Mean', 
        data=fl8_stats,
        color=bar_color,
        capsize=0.1,
        alpha=0.8
    )
    
    # Add error bars (95% confidence interval)
    for i, row in fl8_stats.iterrows():
        ax.errorbar(
            i, row['FL8-A Mean'], 
            yerr=row['FL8-A CI'], 
            fmt='none', 
            c='black', 
            capsize=5
        )
    
    # Adjust x-axis labels if there are many files
    if len(fl8_stats) > 10:
        plt.xticks(rotation=45, ha='right')
    
    # Set chart title and labels
    plt.title('Average FL8-A Values by File', fontsize=14)
    plt.xlabel('File Name', fontsize=12)
    plt.ylabel('FL8-A Mean Value', fontsize=12)
    
    # Add data labels
    for i, v in enumerate(fl8_stats['FL8-A Mean']):
        ax.text(i, v + 0.1, f"{v:.2f}", ha='center', fontsize=9)
    
    # Optimize layout
    plt.tight_layout()
    
    # Save chart (if output folder is specified)
    if output_folder:
        output_path = os.path.join(output_folder, 'FL8-A_by_file.png')
        plt.savefig(output_path, dpi=300)
        print(f"Bar chart saved to: {output_path}")
    
    # Display chart
    plt.show()
    
    # Create logarithmic scale version (if data range is large)
    if fl8_stats['FL8-A Mean'].max() / fl8_stats['FL8-A Mean'].min() > 10:
        plt.figure(figsize=(12, 6))
        
        # Use seaborn to generate bar chart (log scale)
        ax = sns.barplot(
            x='file_name', 
            y='FL8-A Mean', 
            data=fl8_stats,
            color=bar_color,
            capsize=0.1,
            alpha=0.8
        )
        
        # Set logarithmic scale
        plt.yscale('log')
        
        # Format y-axis labels as normal numbers (not scientific notation)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        
        # Adjust x-axis labels if there are many files
        if len(fl8_stats) > 10:
            plt.xticks(rotation=45, ha='right')
        
        # Set chart title and labels
        plt.title('Average FL8-A Values by File (Log Scale)', fontsize=14)
        plt.xlabel('File Name', fontsize=12)
        plt.ylabel('FL8-A Mean Value (Log Scale)', fontsize=12)
        
        # Optimize layout
        plt.tight_layout()
        
        # Save chart (if output folder is specified)
        if output_folder:
            output_path = os.path.join(output_folder, 'FL8-A_by_file_log_scale.png')
            plt.savefig(output_path, dpi=300)
            print(f"Log scale bar chart saved to: {output_path}")
        
        # Display chart
        plt.show()
    
    # Create version sorted by sample size
    plt.figure(figsize=(12, 6))
    
    # Sort by sample size
    fl8_stats_sorted = fl8_stats.sort_values('Sample Size', ascending=False)
    
    # Generate bar chart
    ax = sns.barplot(
        x='file_name', 
        y='FL8-A Mean', 
        data=fl8_stats_sorted,
        order=fl8_stats_sorted['file_name'],
        color=bar_color,
        capsize=0.1,
        alpha=0.8
    )
    
    # Add error bars
    for i, (_, row) in enumerate(fl8_stats_sorted.iterrows()):
        ax.errorbar(
            i, row['FL8-A Mean'], 
            yerr=row['FL8-A CI'], 
            fmt='none', 
            c='black', 
            capsize=5
        )
    
    # Adjust x-axis labels if there are many files
    if len(fl8_stats) > 10:
        plt.xticks(rotation=45, ha='right')
    
    # Set chart title and labels
    plt.title('Average FL8-A Values by File (Sorted by Sample Size)', fontsize=14)
    plt.xlabel('File Name', fontsize=12)
    plt.ylabel('FL8-A Mean Value', fontsize=12)
    
    # Optimize layout
    plt.tight_layout()
    
    # Save chart (if output folder is specified)
    if output_folder:
        output_path = os.path.join(output_folder, 'FL8-A_by_file_sorted.png')
        plt.savefig(output_path, dpi=300)
        print(f"Bar chart sorted by sample size saved to: {output_path}")
    
    # Display chart
    plt.show()
    
    # Output CSV statistics
    if output_folder:
        stats_path = os.path.join(output_folder, 'FL8-A_statistics.csv')
        fl8_stats.to_csv(stats_path, index=False)
        print(f"FL8-A statistics saved to: {stats_path}")
    
    return fl8_stats
