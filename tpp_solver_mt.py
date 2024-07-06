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




if __name__ == "__main__":
    main()