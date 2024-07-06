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

if __name__ == "__main__":
    main()