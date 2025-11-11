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
            df = pd.DataFrame(data, columns=data._channel_labels)
            
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

        