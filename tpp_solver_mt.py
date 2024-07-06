'''
 ,`````.          _________
' AWAU  `,       /_  ___   \
'        `.     /@ \/@  \   \
 ` , . , '  `.. \__/\___/   /
                 \_\/______/
                 /     /\\\\\
                |     |\\\\\\
                 \      \\\\\\
                  \______/\\\\     -ccw-
            _______ ||_||_______
           (______(((_(((______(@)
           Cooked, Fried, and Prepared by Mr Gugi
'''
import numpy as np
import pandas as pd
import streamlit as st


# File reading functions
def read_tsv_file(file_path):
    return pd.read_csv(file_path, sep='\t')

def read_csv_file(file_path):
    return pd.read_csv(file_path)

# Curve fitting functions
def sigmoid(T, a, b, plateau):
    return a / (1 + np.exp(-(T - b))) + plateau

def paper_sigmoidal(T, A1, A2, Tm):
    return A2 + (A1 - A2) / (1 + np.exp((T - Tm)))

# Data extraction and processing functions

# Extract sample information from CSV
def extract_samples(csv_data):
    if "Samples" not in csv_data.columns:
        raise ValueError("The CSV file does not contain a 'Samples' column")
    if "Temperature" not in csv_data.columns:
        raise ValueError("The CSV file does not contain a 'Temperature' column")
    if "Treatment" not in csv_data.columns:
        raise ValueError("The CSV file does not contain a 'Treatment' column")

    return {
        "Temperature": [x for x in csv_data["Temperature"].tolist() if str(x) != 'nan'],
        "Treatment": [x for x in csv_data["Treatment"].tolist() if str(x) != 'nan'],
        "Samples": [x for x in csv_data["Samples"].tolist() if str(x) != 'nan'],
    }

# Filter data and find lowest non-zero float in samples
def filter_and_lowest_float(tsv_data, samples, max_zeroes_allowed):
    missing_cols = set(samples) - set(tsv_data.columns)
    if missing_cols:
        raise ValueError("The TSV file does not contain all values from 'Samples' in the CSV")
    
    selected_data = tsv_data[samples]
    num_data = selected_data.apply(pd.to_numeric, errors='coerce')

    zero_counts = (num_data == 0).sum(axis=1)
    filtered_data = tsv_data[zero_counts < max_zeroes_allowed]

    num_filtered_data = filtered_data[samples].apply(pd.to_numeric, errors='coerce')
    num_filtered_data = num_filtered_data.replace(0, np.nan)

    lowest_val = num_filtered_data.min().min()

    if pd.isna(lowest_val):
        raise ValueError("No valid number found")
    
    return filtered_data, lowest_val

# Count rows with too many zeros
def count_invalid_rows(tsv_data, samples, max_zeros_allowed):
    missing_cols = set(samples) - set(tsv_data.columns)
    if missing_cols:
        raise ValueError("The TSV file does not contain all values from 'Samples' in the CSV")
    
    selected_data = tsv_data[samples]
    num_data = selected_data.apply(pd.to_numeric, errors='coerce')

    zero_counts = (num_data == 0).sum(axis=1)
    excess_zero_rows = (zero_counts >= max_zeros_allowed).sum()

    return excess_zero_rows

# Impute missing data with random values
def impute_filtered_data(filtered_data, samples, lowest_float):
    data_to_process = filtered_data[samples].copy()

    for col in samples:
        num_col = pd.to_numeric(data_to_process[col], errors='coerce')
        zero_mask = num_col == 0
        zero_count = zero_mask.count()

        if zero_count > 0:
            rand_vals = np.random.uniform(0, lowest_float, size=zero_count)
            num_col[zero_mask] = rand_vals
        
        data_to_process[col] = num_col

    filtered_data[samples] = data_to_process

    return filtered_data

# Group samples by treatment and temperature
def get_replicant_lists(csv_data):
    if not all(col in csv_data.columns for col in ['Samples', 'Temperature', 'Treatment']):
        raise ValueError("Does not contain appropriate columns")
    
    csv_data['Temperature'] = pd.to_numeric(csv_data['Temperature'], errors='coerce')
    csv_data_sorted = csv_data.sort_values(['Treatment', 'Temperature'])

    grouped = csv_data_sorted.groupby(['Treatment', 'Temperature'])
    sample_groups = {}
    for (treatment, temp), group in grouped:
        if treatment not in sample_groups:
            sample_groups[treatment] = []
        sample_groups[treatment].append((temp, group['Samples'].tolist()))

    return sample_groups

# Calculate average values for each group
def average_samples(sample_groups, filtered_data):
    averaged_data = {}
    for treatment in sample_groups:

        if treatment not in averaged_data:
            averaged_data[treatment] = {}

        for sample in sample_groups[treatment]:
            for _, row in filtered_data.iterrows():
                protein_id = row['Protein ID']

                sample_vals = row[sample[1]]
                average_val = np.mean(sample_vals)

                if protein_id not in averaged_data[treatment]:
                    averaged_data[treatment][protein_id] = {}

                averaged_data[treatment][protein_id][sample[0]] = (average_val)
    return averaged_data

def main():
    # Set up Streamlit interface
    st.title("TPP Analysis App")

    # Handle file uploads
    tsv_file = st.file_uploader("Upload TSV raw data file", type=['tsv'])
    csv_file = st.file_uploader("Upload CSV metadata file", type=['csv'])

    max_allowed_zeros = st.number_input("Maximum number of zeros allowed", min_value=0, value=20, step=1)

    if tsv_file and csv_file:

        # Process data
        tsv_data = read_tsv_file(tsv_file)
        csv_data = read_csv_file(csv_file)
        metadata = extract_samples(csv_data)
        
        droppable_rows = count_invalid_rows(tsv_data, metadata["Samples"], max_allowed_zeros)
        st.subheader(f"Number of rows with {max_allowed_zeros} or more zeros: {droppable_rows} (Dropped)")

        col1, col2 = st.columns(2)
        with col1:
            cont_btn = st.button("Continue Analysis")
        with col2:
            stop_btn = st.button("Stop Analysis")

        if cont_btn:    
            # Generate and display results
            filtered_data, ceiling_rand = filter_and_lowest_float(tsv_data,metadata['Samples'],max_allowed_zeros)

            st.subheader("Analysis Result")
            st.write(f"Number of rows after filtering: {len(filtered_data)}")
            st.write(f"Number of rows removed: {len(tsv_data) - len(filtered_data)}")
            st.write(f"Lowest non-zero float number found (after filtering): {ceiling_rand}")

            filtered_data_imputed = impute_filtered_data(filtered_data.copy(), metadata['Samples'], ceiling_rand)
            sample_groups = get_replicant_lists(csv_data)
            average_dict = average_samples(sample_groups, filtered_data_imputed)




if __name__ == "__main__":
    main()