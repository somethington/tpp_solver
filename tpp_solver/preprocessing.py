"""Pure data preparation: filtering, imputation, reshaping and summaries."""
import numpy as np
import pandas as pd

def filter_and_lowest_float(tsv_data, samples):
    missing_cols = set(samples) - set(tsv_data.columns)
    if missing_cols:
        raise ValueError("The TSV file does not contain all values from 'Samples' in the CSV")
    
    filtered_data = tsv_data  # No filtering based on zeros anymore

    num_filtered_data = filtered_data[samples].apply(pd.to_numeric, errors='coerce')
    num_filtered_data = num_filtered_data.replace(0, np.nan)

    lowest_val = num_filtered_data.min().min()

    if pd.isna(lowest_val):
        raise ValueError("No valid number found")
    
    return filtered_data, lowest_val

def impute_filtered_data(filtered_data, samples, lowest_float, seed=None):
    """Impute zeros with small random values drawn below the lowest observed value.

    A ``seed`` makes the imputation (and therefore the downstream melting points)
    reproducible across runs.
    """
    rng = np.random.default_rng(seed)
    data_to_process = filtered_data[samples].copy()

    for col in samples:
        num_col = pd.to_numeric(data_to_process[col], errors='coerce')
        zero_mask = num_col == 0
        zero_count = int(zero_mask.sum())

        if zero_count > 0:
            rand_vals = rng.uniform(0, lowest_float, size=zero_count)
            num_col[zero_mask] = rand_vals

        data_to_process[col] = num_col

    filtered_data[samples] = data_to_process

    return filtered_data

def get_replicate_data(csv_data, filtered_data):
    """
    Memory-efficient organization of data by treatment, temperature, and replicate.
    
    Args:
        csv_data (pd.DataFrame): DataFrame containing metadata with Temperature, Treatment, Samples columns
        filtered_data (pd.DataFrame): Filtered protein intensity data
    
    Returns:
        dict: Nested dictionary with format {treatment: {protein: {temp: list_of_replicate_values}}}
    """
    # First, create an efficient mapping of treatment/temperature combinations to their samples
    treatment_temp_map = {}
    for treatment in csv_data['Treatment'].unique():
        treatment_data = csv_data[csv_data['Treatment'] == treatment]
        for temp in treatment_data['Temperature'].unique():
            samples = treatment_data[treatment_data['Temperature'] == temp]['Samples'].tolist()
            if treatment not in treatment_temp_map:
                treatment_temp_map[treatment] = {}
            treatment_temp_map[treatment][float(temp)] = samples

    # Process proteins one at a time
    replicate_data = {}
    
    # Process filtered_data in chunks to reduce memory usage
    chunk_size = 100  # Adjust this based on your available memory
    
    for chunk_start in range(0, len(filtered_data), chunk_size):
        chunk = filtered_data.iloc[chunk_start:chunk_start + chunk_size]
        
        for _, row in chunk.iterrows():
            protein_id = row['Protein ID']
            
            # Process each treatment
            for treatment, temp_data in treatment_temp_map.items():
                if treatment not in replicate_data:
                    replicate_data[treatment] = {}
                if protein_id not in replicate_data[treatment]:
                    replicate_data[treatment][protein_id] = {}
                
                # Process each temperature
                for temp, samples in temp_data.items():
                    if temp not in replicate_data[treatment][protein_id]:
                        replicate_data[treatment][protein_id][temp] = []
                    
                    # Get intensity values for all replicates at this temperature
                    try:
                        intensities = [float(row[sample]) for sample in samples]
                        replicate_data[treatment][protein_id][temp] = intensities
                    except (ValueError, KeyError):
                        continue
    
    return replicate_data

def _slice_replicate_data(replicate_data, protein):
    """Extract a single protein's data so workers receive only what they need.

    Sending the full ``replicate_data`` to every worker pickles the entire
    dataset once per protein, which dominates runtime and memory on large
    proteomes. This returns the same nested shape but for one protein only.
    """
    return {
        treatment: {protein: proteins[protein]}
        for treatment, proteins in replicate_data.items()
        if protein in proteins
    }

def calculate_bin_width(data):
    data = data.dropna()  # Remove NaN values
    q25, q75 = np.percentile(data, [25, 75])  # Calculate the 25th and 75th percentiles
    iqr = q75 - q25  # Interquartile range
    bin_width = 2 * iqr * len(data) ** (-1/3)  # Freedman-Diaconis rule
    bin_width = max(bin_width, 1e-3)  # Ensure bin width is not too small
    bins = int(np.ceil((data.max() - data.min()) / bin_width))  # Number of bins
    return max(bins, 10)  # Ensure at least 10 bins

def process_summary_tables(summary_table):
    """
    Generate both replicate and averaged summary tables from the original data.
    
    Args:
        summary_table (pd.DataFrame): Original summary table
        
    Returns:
        tuple: (replicate_table, averaged_table) where averaged_table contains mean values
              for each protein-treatment combination
    """
    # Copy original table
    replicate_table = summary_table.copy()
    
    # Get available columns
    available_columns = summary_table.columns.tolist()
    
    # Define base columns that should be in both tables
    base_cols = ['protein', 'treatment', 'melting_point', 'R²', 'residuals']
    
    # Add optional columns if they exist
    optional_cols = ['Protein Name', 'GO ID', 'Function', 'Link']
    for col in optional_cols:
        if col in available_columns:
            base_cols.append(col)
    
    # Calculate average values
    agg_dict = {
        'melting_point': 'mean',
        'R²': 'mean',
        'residuals': lambda x: x.iloc[0]  # Keep first residual
    }
    
    # Add optional columns to aggregation
    for col in optional_cols:
        if col in available_columns:
            agg_dict[col] = 'first'
    
    # Create averaged table
    averaged_table = (summary_table.groupby(['protein', 'treatment'])
                     .agg(agg_dict)
                     .reset_index())
    
    # Add standard deviation and number of replicates
    std_dev = (summary_table.groupby(['protein', 'treatment'])
               ['melting_point']
               .std()
               .reset_index()
               .rename(columns={'melting_point': 'melting_point_std'}))
    
    replicate_count = (summary_table.groupby(['protein', 'treatment'])
                      .size()
                      .reset_index()
                      .rename(columns={0: 'num_replicates'}))
    
    # Merge additional statistics into averaged table
    averaged_table = (averaged_table.merge(std_dev, on=['protein', 'treatment'], how='left')
                     .merge(replicate_count, on=['protein', 'treatment'], how='left'))
    
    # Fill NaN values in standard deviation
    averaged_table['melting_point_std'] = averaged_table['melting_point_std'].fillna(0)
    
    # Define column orders
    averaged_cols = ['protein', 'treatment', 'melting_point', 'melting_point_std',
                    'num_replicates', 'R²', 'residuals']
    
    # Add optional columns to order if they exist
    for col in optional_cols:
        if col in averaged_table.columns:
            averaged_cols.append(col)
    
    # Ensure columns exist before reordering
    averaged_table = averaged_table[[col for col in averaged_cols if col in averaged_table.columns]]
    replicate_table = replicate_table[[col for col in base_cols if col in replicate_table.columns]]
    
    return replicate_table, averaged_table
