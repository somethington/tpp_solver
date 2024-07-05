import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import shutil
import os
import time
import streamlit as st

def upload():
    # Upload the data and metadata files. 
    uploaded_file = st.file_uploader("Upload TSV file", type="tsv")
    metadata = st.file_uploader("Upload Metadata tsv", type="tsv")
    if uploaded_file is not None and metadata is not None:
        df = pd.read_csv(uploaded_file, delimiter='\t')
        metadata = pd.read_csv(metadata, delimiter='\t')
    else:  
        st.write("Please upload a file") 
        st.stop()
    return df, metadata

def extract_metadata(metadata):
    # Extract the metadata from the metadata file
    temperature = metadata['temperature'].values
    datapoints = len(temperature)
    return temperature, datapoints

def get_num_replicates(control_prefix,treatment_prefix, protein_dataframes):
    # Find the number of replicates based on this pattern: prefix_number
    # iterate over columns and find the biggest value for prefix_number
    protein = list(protein_dataframes.keys())[0]
    columns = protein_dataframes[protein].columns
    control_columns = [col for col in columns if control_prefix in col]
    treatment_columns = [col for col in columns if treatment_prefix in col]
    control_numbers = set()
    treatment_numbers = set()
    for col in control_columns:
        number = int(col.split('_')[1])
        control_numbers.add(number)
    for col in treatment_columns:
        number = int(col.split('_')[1])
        treatment_numbers.add(number)
    max_control_number = max(control_numbers)
    max_treatment_number = max(treatment_numbers)

    return max_control_number, max_treatment_number

def get_prefixes():
    # Ask user to input prefix for control and treatment columns
    control_prefix, treatment_prefix = st.columns(2)
    with control_prefix:
        control_prefix = st.text_input("Enter the prefix for control columns")
    with treatment_prefix:    
        treatment_prefix = st.text_input("Enter the prefix for treatment columns")
    if control_prefix is None and treatment_prefix is None:
        st.write("Please enter the prefixes")
        st.stop()
        
    return control_prefix, treatment_prefix

def download(figures_dir):
    shutil.make_archive(figures_dir, 'zip', figures_dir)

    with open(figures_dir + ".zip", "rb") as file:
        st.download_button(
            label="Download Output",
            data=file,
            file_name="output.zip",
            mime="application/zip",
        )
def column_selector(df):
    options = ['MaxLFQ', 'Intensity']

    intensity = st.selectbox("Select a column",options, index=None, placeholder='Select a column')
    if intensity == options[0]:
        intensity = [col for col in df.columns if options[0] in col]
    elif intensity == options[1]:
        intensity = [col for col in df.columns if options[1] in col and options[0] not in col]
    else:
        st.write("Please select correct column")
        st.stop()
    return intensity

def protein_df_filler(df, intensity):
    # Fill the protein dataframes dictionary with the dataframes for each protein
    # Finds lowest nonzero value
    protein_dataframes = {}
    lowest_nonzero = df[intensity].replace(0, np.nan).min().min()
    for protein_id in df['Protein ID'].unique():
        protein_df = df[df['Protein ID'] == protein_id][intensity]
        protein_dataframes[protein_id] = protein_df

    return protein_dataframes, lowest_nonzero

def set_zero_limit():
    # Ask the user to set a limit for number of zeros in the data
    zero_limit = st.number_input('Set a limit for number of zeros in the data', min_value=0, max_value=100)
    if zero_limit is None:
        st.write("Please enter a value")
        st.stop()
    return zero_limit

def drop_by_zero_limit(protein_dataframes, zero_limit):
    # Drop the proteins with more than the zero_limit number of zeros
    keys_to_drop = []
    for protein in protein_dataframes.keys():
        zero_count = (protein_dataframes[protein] == 0).sum().sum()
        if zero_count > zero_limit:
            keys_to_drop.append(protein)

    num_keys_dropped = len(keys_to_drop)
    st.write(f"Number of samples that will be dropped: {num_keys_dropped} out of {len(protein_dataframes.keys())}")

    if st.button("Drop samples under the zero limit"):
        for key in keys_to_drop:
            protein_dataframes.pop(key)
        st.write("Samples dropped")

    return protein_dataframes   

def mean_intensity(protein, control_prefix, treatment_prefix, max_control_number, max_treatment_number):
    treatment_dataframes = {}
    control_dataframes = {}
    control_prefix_lst = [f"{control_prefix}_{i}" for i in range(1, max_control_number + 1)]
    treatment_prefix_lst = [f"{treatment_prefix}_{i}" for i in range(1, max_treatment_number + 1)]
    for prefix in treatment_prefix_lst:
        selected_columns = [col for col in protein.columns if prefix in col]
        treatment_dataframes[prefix] = protein[selected_columns].values.flatten().tolist()

    for prefix in control_prefix_lst:
        selected_columns = [col for col in protein.columns if prefix in col]
        control_dataframes[prefix] = protein[selected_columns].values.flatten().tolist()
    
    treatment_mean = [sum(values) / len(values) for values in zip(*treatment_dataframes.values())]
    control_mean = [sum(values) / len(values) for values in zip(*control_dataframes.values())]

    return control_mean, treatment_mean


def impute(protein_dataframes, lowest_nonzero):
    # Replace zeros with random values between 0 and the lowest nonzero value
    # iterate over each column in each protein dataframe and replace zeros with random values
    for protein in protein_dataframes.keys():
        protein_dataframes[protein] = protein_dataframes[protein].map(lambda x: np.random.uniform(0, lowest_nonzero) if x == 0 else x)
    return protein_dataframes

def sigmoid(x, a, b, c):
    #return 1 / (1 + np.exp(-A * (Td - T) / T))
    #return (1 - p)/(1+ np.exp(-1*(a/t-b))+p)

    return a / (1 + np.exp(-(x - b))) + c
def normalize(mean_intensity):
    # Normalize by iterating over each value in mean_intensity list and dividing it by the first value
    normalized_mean = [entry / mean_intensity[0] for entry in mean_intensity]

    return normalized_mean

def get_residuals(c_1, t_1, popt_c1, popt_t1,temperature):
    residuals_c = c_1 - sigmoid(temperature, *popt_c1)
    residuals_t = t_1 - sigmoid(temperature, *popt_t1)
    return residuals_c, residuals_t

def tpp_fitting(c_1, t_1, temperature,protein_id):

    # Perform sigmoidal fitting for c_1
    popt_c1, pcov_c1 = curve_fit(sigmoid, temperature, c_1, p0=[max(c_1), np.median(temperature), min(c_1)], method='dogbox')

    # Generate fitted curve for c_1
    x_fit_c1 = np.linspace(min(temperature), max(temperature), 100)
    y_fit_c1 = sigmoid(x_fit_c1, *popt_c1)

    # Perform sigmoidal fitting for t_1
    popt_t1, pcov_t1 = curve_fit(sigmoid, temperature, t_1, p0=[max(t_1), np.median(temperature), min(t_1)], method='dogbox')


    # Generate fitted curve for t_1
    x_fit_t1 = np.linspace(min(temperature), max(temperature), 100)
    y_fit_t1 = sigmoid(x_fit_t1, *popt_t1)
    y_point_c1 = sigmoid(popt_c1[1], *popt_c1)
    y_point_t1 = sigmoid(popt_t1[1], *popt_t1)
    # Calculate residuals

    residuals_c, residuals_t = get_residuals(c_1, t_1, popt_c1, popt_t1, temperature)
    # Calculate the delta between the two points
    summary_table.loc[len(summary_table)] = {
        'protein_id': protein_id,
        'control melting point': popt_c1[1],
        'treatment melting point': popt_t1[1],
        'residuals_c': ','.join(map(str, residuals_c)),
        'residuals_t': ','.join(map(str, residuals_t)),
        'delta': popt_c1[1] - popt_t1[1]
    }
    # Plot the two points using respective popt and y point
    plt.scatter(popt_c1[1], y_point_c1, color='red', label=f'{popt_c1[1]:.3f}')
    plt.scatter(popt_t1[1], y_point_t1, color='green', label=f'{popt_t1[1]:.3f}')
    # Draw a dotted line between the two points
    plt.plot([popt_c1[1], popt_t1[1]], [y_point_c1, y_point_t1], 'k--', linewidth=0.5)
    # Plot the data and fitted curves

    plt.scatter(temperature, c_1, color='orange', marker='s', label='Control Data')
    plt.plot(x_fit_c1, y_fit_c1, 'orange', label='Control fitted curve')
    plt.scatter(temperature, t_1, color='blue', marker='^', label='Treatment Data')
    plt.plot(x_fit_t1, y_fit_t1, 'blue', label='Treatment fitted curve')
    plt.xlabel('Temperature')
    plt.ylabel('Intensity')
    delta_rounded = round(popt_t1[1] - popt_c1[1], 4)
    r_squared_c1 = r2_score(c_1, sigmoid(temperature, *popt_c1))
    r_squared_t1 = r2_score(t_1, sigmoid(temperature, *popt_t1))

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2, title='Δ = {delta}, R^2_c = {r2_c}, R^2_t = {r2_t}'.format(delta=delta_rounded, r2_c=round(r_squared_c1,4), r2_t=round(r_squared_t1,4)))
    plt.tight_layout()
    plt.savefig('/Users/lukasdolidze/Coding/toolbox/toolbox/figures/{protein_id}.svg'.format(protein_id = protein_id))
    plt.close()

    

summary_table = pd.DataFrame(columns=['protein_id', 'control melting point', 'treatment melting point', 'residuals_c', 'residuals_t', 'delta'])
    

def run_loop(protein_dataframes, control_prefix, treatment_prefix, max_control_number, max_treatment_number,temperature,start_time,figures_dir):
    print(len(protein_dataframes))
    if st.button('Run'):
        print(len(protein_dataframes))
        with st.spinner('Running...'):
            for protein in protein_dataframes.keys():
                   st.write(len(protein_dataframes))
                   control_mean, treatment_mean = mean_intensity(protein_dataframes[protein],
                    control_prefix, treatment_prefix, max_control_number, max_treatment_number)
                   control_mean = normalize(control_mean)
                   treatment_mean = normalize(treatment_mean)
                   try:
                        tpp_fitting(control_mean, treatment_mean, temperature, protein)
                   except RuntimeError as e:
                        if str(e) == "Optimal parameters not found: ":
                            continue
            if download(figures_dir):
                clean_up(start_time,figures_dir)

def clean_up(start_time,figures_dir):
    # Clean up the figures directory
    shutil.rmtree(figures_dir)
    # Calculate the runtime
    end_time = time.time()
    runtime = end_time - start_time
    st.write(f"Runtime: {runtime} seconds")


def main():
    start_time = time.time()
    # Set the source directory
    source_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the figures directory
    figures_dir = os.path.join(source_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    df, metadata = upload()
    intensity = column_selector(df)
    control_prefix, treatment_prefix = get_prefixes()
    protein_dataframes, lowest_nonzero = protein_df_filler(df, intensity)
    temperature, datapoints = extract_metadata(metadata)
    zero_limit = set_zero_limit()
    protein_dataframes = drop_by_zero_limit(protein_dataframes, zero_limit)
    protein_dataframes = impute(protein_dataframes, lowest_nonzero)
    st.write(len(protein_dataframes))
    max_control_number, max_treatment_number = get_num_replicates(control_prefix, treatment_prefix, protein_dataframes)

    run_loop(protein_dataframes, control_prefix, treatment_prefix, max_control_number, max_treatment_number,temperature,start_time,figures_dir) 
    st.stop()
    
if __name__ == "__main__":
    main()