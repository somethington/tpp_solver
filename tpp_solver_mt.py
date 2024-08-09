import io
import os
import time
import zipfile 
import itertools
import numpy as np
import pandas as pd
import streamlit as st
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from readme_content import  display_readme




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
    # Filter out unnamed columns
    named_columns = [col for col in csv_data.columns if not col.startswith('Unnamed:')]

    # Define the required categories
    categories = ["Temperature", "Treatment", "Samples"]

    # Create a dictionary to store the selected columns for each category
    selected_columns = {}

    for category in categories:
        default_column = category if category in named_columns else None
        selected_columns[category] = st.selectbox(
            f"Select column for {category}:",
            options=[""] + named_columns,
            index=named_columns.index(default_column) + 1 if default_column else 0
        )

    result = {}
    for category, column in selected_columns.items():
        if not column:
            st.error(f"Please select a column for {category}")
            return None
        if column not in csv_data.columns:
            st.error(f"The CSV file does not contain a '{column}' column")
            return None
        result[category] = [x for x in csv_data[column].tolist() if str(x) != 'nan']
    
    return {
        "Temperature": result.get("Temperature", []),
        "Treatment": result.get("Treatment", []),
        "Samples": result.get("Samples", [])
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
        zero_count = zero_mask.sum()

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

                averaged_data[treatment][protein_id][sample[0]] = average_val
    return averaged_data

# Fit curve and plot data for a single protein
def process_protein(args):
    protein, data_dict, markers, sizes, alphas, positions = args
    fig, ax = plt.subplots(figsize=(10,6))
    summary_data = []
    
    try:
        for treatment, proteins in data_dict.items():
            if protein in proteins:
                temperatures = np.array(list(proteins[protein].keys()))
                values = np.array(list(proteins[protein].values()))

                index = np.argsort(temperatures)
                temperatures = temperatures[index]
                values = values[index]

                values = [entry / values[0] for entry in values]

                valmax = max(values)
                med = np.median(temperatures)
                minval = min(values)

                marker = next(markers)
                size = next(sizes)
                alpha = next(alphas)
                curpos = next(positions)

                ax.scatter(temperatures, values, marker=marker, s=size, alpha=alpha, label=f'{protein} {treatment} Curve')

                popt, _ = curve_fit(sigmoid, temperatures, values, p0=[valmax, med, minval])

                melt_pt = sigmoid(popt[1], *popt)

                temp_range = np.linspace(temperatures.min(), temperatures.max(), 100)
                ax.plot(temp_range, sigmoid(temp_range, *popt), '--', alpha=0.7, label=f'{protein} {treatment} Fitted')

                ax.scatter(popt[1], melt_pt, color='red', s=75, marker='^')
                ax.text(popt[1], melt_pt, f'{popt[1]:.2f}', color='red', horizontalalignment=curpos, verticalalignment='bottom')

                summary_data.append({
                    'protein_id': protein,
                    'treatment': treatment,
                    'melting point': popt[1],
                    'residuals': ','.join(map(str, (values - sigmoid(temperatures, *popt)))),
                })

        ax.set_xlabel('Temperature')                        
        ax.set_ylabel('Intensity')
        ax.set_title(f'Fitted Curve for {protein}')
        ax.legend()
        
        return protein, fig, summary_data
    except RuntimeError as e:
        print(f"Error processing protein {protein}: {str(e)}")
        return None, None, None
    except Exception as e:
        print(f"Unexpected error processing protein {protein}: {str(e)}")
        return None, None, None

# Fit curves and plot data for all proteins using multiprocessing
def fit_and_plot(data_dict):
    all_proteins = set()
    for treatment in data_dict.values():
        all_proteins.update(treatment.keys())
    
    markers = itertools.cycle(['o', 's'])
    sizes = itertools.cycle([50, 100])
    alphas = itertools.cycle([1.0, 0.5])
    positions = itertools.cycle(["left", "right"])

    pool = mp.Pool(processes=mp.cpu_count())
    results = pool.map(process_protein, [(protein, data_dict, markers, sizes, alphas, positions) for protein in all_proteins])
    pool.close()
    pool.join()

    figures = {}
    all_summary_data = []

    for result in results:
        if result is not None:
            protein, fig, summary_data = result
            if protein is not None and summary_data:
                figures[protein] = fig
                all_summary_data.extend(summary_data)

    # Create the summary table only if we have data
    if all_summary_data:
        summary_table = pd.DataFrame(all_summary_data)
        # Ensure correct data types
        summary_table['protein_id'] = summary_table['protein_id'].astype(str)
        summary_table['treatment'] = summary_table['treatment'].astype(str)
        summary_table['melting point'] = summary_table['melting point'].astype(float)
        summary_table['residuals'] = summary_table['residuals'].astype(str)
    else:
        summary_table = pd.DataFrame(columns=['protein_id', 'treatment', 'melting point', 'residuals'])

    return figures, summary_table

# Save a single figure as SVG
def save_figure_as_svg(args):
    protein, fig = args
    svg_io = io.BytesIO()
    fig.savefig(svg_io, format='svg', bbox_inches='tight')
    svg_io.seek(0)
    plt.close(fig)
    return (f"{protein}.svg", svg_io.getvalue())

# Save all figures as SVGs and create a zip file
def save_as_svg(figures, dataframe):
    pool = mp.Pool(processes=mp.cpu_count())
    memory_files = pool.map(save_figure_as_svg, figures.items())
    pool.close()
    pool.join()

    csv_io = io.BytesIO()
    dataframe.to_csv(csv_io, index=False)
    csv_io.seek(0)
    memory_files.append(("summary.csv", csv_io.getvalue()))

    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, mode='w') as zip_file:
        for file_name, file_content in memory_files:
            zip_file.writestr(file_name, file_content)

    return zip_io.getvalue()

sample_help = "By pressing this button, sample experimental data will be loaded for demonstration purposes"

def analysis():
    st.title("TPP Analysis App")

    # Initialize session state variables
    if 'tsv_data' not in st.session_state:
        st.session_state.tsv_data = None
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = None
    if 'edit_mode_tsv' not in st.session_state:
        st.session_state.edit_mode_tsv = False
    if 'edit_mode_csv' not in st.session_state:
        st.session_state.edit_mode_csv = False

    # File uploaders
    uploaded_tsv = st.file_uploader("Upload TSV fragpipe output file", type=['tsv'])
    uploaded_csv = st.file_uploader("Upload CSV metadata file", type=['csv'])

    # Create a row with three columns for the buttons
    col1, col2, col3 = st.columns(3)

    # Button to load uploaded data
    with col1:
        if st.button('Load uploaded data'):
            if uploaded_tsv is not None and uploaded_csv is not None:
                st.session_state.tsv_data = read_tsv_file(uploaded_tsv)
                st.session_state.csv_data = read_csv_file(uploaded_csv)
                st.success("Uploaded data loaded successfully!")
            else:
                st.warning("Please upload both TSV and CSV files before loading.")

    # Button to load sample data
    with col2:
        if st.button('Load sample data', help=sample_help):
            data_path = os.path.dirname(os.path.abspath(__file__))
            tsv_path = os.path.join(data_path, "sample_data.tsv")
            csv_path = os.path.join(data_path, "sample_metadata.csv")
            st.session_state.tsv_data = read_tsv_file(tsv_path)
            st.session_state.csv_data = read_csv_file(csv_path)
            st.success("Sample data loaded successfully!")

    # Button to clear loaded data
    with col3:
        if st.button('Clear loaded data'):
            st.session_state.tsv_data = None
            st.session_state.csv_data = None
            st.session_state.edit_mode_tsv = False
            st.session_state.edit_mode_csv = False
            st.success("All loaded data has been cleared!")

    # Display data status
    if st.session_state.tsv_data is not None and st.session_state.csv_data is not None:
        st.info("TSV and CSV data are loaded and ready for analysis.")
    else:
        st.warning("Please load both TSV and CSV data to proceed with analysis.")

    # Display and edit TSV data
    if st.session_state.tsv_data is not None:
        with st.expander("TSV Data", expanded=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("TSV Data")
            with col2:
                edit_button = st.button("Toggle Edit Mode (TSV)")
            
            if edit_button:
                st.session_state.edit_mode_tsv = not st.session_state.edit_mode_tsv

            if st.session_state.edit_mode_tsv:
                edited_tsv = st.data_editor(st.session_state.tsv_data, num_rows="dynamic")
                if st.button("Save TSV Changes"):
                    st.session_state.tsv_data = edited_tsv
                    st.success("TSV data changes saved!")
            else:
                st.dataframe(st.session_state.tsv_data)

    # Display and edit CSV data
    if st.session_state.csv_data is not None:
        with st.expander("CSV Metadata", expanded=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("CSV Metadata")
            with col2:
                edit_button = st.button("Toggle Edit Mode (CSV)")
            
            if edit_button:
                st.session_state.edit_mode_csv = not st.session_state.edit_mode_csv

            if st.session_state.edit_mode_csv:
                edited_csv = st.data_editor(st.session_state.csv_data, num_rows="dynamic")
                if st.button("Save CSV Changes"):
                    st.session_state.csv_data = edited_csv
                    st.success("CSV metadata changes saved!")
            else:
                st.dataframe(st.session_state.csv_data)

    if st.session_state.tsv_data is not None and st.session_state.csv_data is not None:
            st.subheader("Analysis Setup")

            # Extract sample information from CSV
            metadata = extract_samples(st.session_state.csv_data)

            if metadata is None:
                st.error("Failed to extract samples from metadata. Please check your CSV file.")
            else:
                # Set maximum number of zeros allowed
                max_allowed_zeros = st.number_input("Maximum number of zeros allowed", min_value=0, value=20, step=1)

                # Count rows to be dropped
                droppable_rows = count_invalid_rows(st.session_state.tsv_data, metadata["Samples"], max_allowed_zeros)
                st.write(f"Number of rows with {max_allowed_zeros} or more zeros: {droppable_rows} (Will be dropped)")

                # Start Analysis button
                if st.button("Start Analysis"):
                    # Process data
                    tsv_data = st.session_state.tsv_data
                    csv_data = st.session_state.csv_data

                    # Generate and display results
                    filtered_data, ceiling_rand = filter_and_lowest_float(tsv_data, metadata['Samples'], max_allowed_zeros)

                    st.subheader("Analysis Results")
                    st.write(f"Number of rows after filtering: {len(filtered_data)}")
                    st.write(f"Number of rows removed: {len(tsv_data) - len(filtered_data)}")
                    st.write(f"Lowest non-zero float number found (after filtering): {ceiling_rand}")

                    filtered_data_imputed = impute_filtered_data(filtered_data.copy(), metadata['Samples'], ceiling_rand)
                    sample_groups = get_replicant_lists(csv_data)
                    average_dict = average_samples(sample_groups, filtered_data_imputed)

                    start_time = time.time()
                    with st.spinner("Fitting curves and generating plots..."):
                        figures, summary_table = fit_and_plot(average_dict)
                    end_time = time.time()
                    figure_generation_time = end_time - start_time

                    st.write(f"Time taken to generate figures: {figure_generation_time:.2f} seconds")

                    start_time = time.time()
                    with st.spinner('Preparing SVG files for download...'):
                        zip_file = save_as_svg(figures, summary_table)
                    end_time = time.time()
                    figure_save_time = end_time - start_time

                    st.write(f"Time taken to save figures: {figure_save_time:.2f} seconds")

                    # Provide download option for results
                    st.download_button(
                        label="Download zipped SVGs",
                        data=zip_file,
                        file_name="protein_curves.zip",
                        mime="application/zip"
                    )

                    # Display summary table
                    st.subheader("Summary Table")
                    st.dataframe(summary_table)

                    plt.close('all')
def main():

    st.set_page_config(
    page_title="TPP Solver",
    page_icon="logo_32x32.png",  
    layout="wide",
)
    st.sidebar.title("Navigation")
    
    # Sidebar navigation
    page = st.sidebar.radio("Go to", ["Main App", "README"])

    if page == "README":
        display_readme()
    elif page == "Main App":
        analysis()

if __name__ == "__main__":
    main()