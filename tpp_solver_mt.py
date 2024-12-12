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
import duckdb
from scipy.optimize import curve_fit
from scipy.stats import shapiro, boxcox, yeojohnson, gaussian_kde, mannwhitneyu
from readme_content import display_readme
import plotly.graph_objects as go
import plotly.express as px
import textwrap


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
def extract_samples(csv_data):
    named_columns = [col for col in csv_data.columns if not col.startswith('Unnamed:')]
    categories = ["Temperature", "Treatment", "Samples"]
    selected_columns = {}

    for category in categories:
        default_column = category if category in named_columns else None
        selected_columns[category] = st.selectbox(
            f"Select column for {category}:",
            options=[""] + named_columns,
            index=named_columns.index(default_column) + 1 if default_column else 0,
            key=f"{category}_column"
        )

    result = {}
    for category, column in selected_columns.items():
        if not column:
            st.error(f"Please select a column for {category}")
            return None
        if column not in csv_data.columns:
            st.error(f"The CSV file does not contain a '{column}' column")
            return None
        result[category] = csv_data[column].dropna().tolist()
    
    return {
        "Temperature": result.get("Temperature", []),
        "Treatment": result.get("Treatment", []),
        "Samples": result.get("Samples", [])
    }

# Filter data and find lowest non-zero float in samples
def filter_and_lowest_float(tsv_data, samples):
    missing_cols = set(samples) - set(tsv_data.columns)
    if missing_cols:
        raise ValueError("The TSV file does not contain all values from 'Samples' in the CSV")
    
    selected_data = tsv_data[samples]
    num_data = selected_data.apply(pd.to_numeric, errors='coerce')
    filtered_data = tsv_data  # No filtering based on zeros anymore

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
        raise ValueError("The TSV file does not contain all values from 'Samples' in the CSV" , missing_cols)
    
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


def process_protein_replicates(args):
    """
    Process individual protein replicates for curve fitting with memory efficiency.
    
    Returns:
        tuple: (protein, fig, summary_data) if successful, (None, None, None) if not
    """
    (
        protein,
        data_dict,
        markers,
        sizes,
        alphas,
        positions,
        selected_temp,
        normalize_data,
        r2_threshold,
    ) = args
    
    summary_data = []
    fig = None
    ax = None
    
    try:
        y_offset = 0.0
        for treatment, proteins in data_dict.items():
            if protein in proteins:
                protein_data = proteins[protein]
                
                # Get number of replicates from first temperature point
                first_temp = list(protein_data.keys())[0]
                num_replicates = len(protein_data[first_temp])
                
                replicate_fits = []
                
                # Process each replicate
                for replicate_idx in range(num_replicates):
                    temperatures = []
                    values = []
                    
                    # Collect data points for this replicate
                    for temp in sorted(protein_data.keys()):
                        if len(protein_data[temp]) > replicate_idx:
                            temperatures.append(temp)
                            values.append(protein_data[temp][replicate_idx])
                    
                    if len(temperatures) < 4:  # Need minimum points for fitting
                        continue
                        
                    temperatures = np.array(temperatures)
                    values = np.array(values)
                    
                    if normalize_data and selected_temp in temperatures:
                        norm_idx = np.where(temperatures == selected_temp)[0][0]
                        norm_value = values[norm_idx]
                        if norm_value != 0:  # Avoid division by zero
                            values = values / norm_value
                    
                    try:
                        # Fit sigmoid curve
                        valmax = max(values)
                        med = np.median(temperatures)
                        minval = min(values)
                        
                        popt, _ = curve_fit(sigmoid, temperatures, values, p0=[valmax, med, minval])
                        fitted_values = sigmoid(temperatures, *popt)
                        
                        # Calculate R²
                        ss_res = np.sum((values - fitted_values) ** 2)
                        ss_tot = np.sum((values - np.mean(values)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        
                        if r_squared >= r2_threshold:
                            replicate_fits.append({
                                'temperatures': temperatures,
                                'values': values,
                                'popt': popt,
                                'r_squared': r_squared,
                                'fitted_values': fitted_values,
                                'replicate_num': replicate_idx
                            })
                            
                            summary_data.append({
                                'protein': protein,
                                'treatment': treatment,
                                'replicate': replicate_idx,
                                'melting_point': popt[1],
                                'R²': r_squared,
                                'residuals': ','.join(map(str, (values - fitted_values)))
                            })
                            
                    except RuntimeError:
                        continue
                
                # Create plot if we have any successful fits
                if replicate_fits:
                    if fig is None:
                        fig, ax = plt.subplots(figsize=(10, 6))
                    
                    marker = next(markers)
                    size = next(sizes)
                    alpha = next(alphas)
                    curpos = next(positions)
                    
                    # Plot each replicate
                    for i, fit_data in enumerate(replicate_fits):
                        # Plot measured points
                        ax.scatter(
                            fit_data['temperatures'],
                            fit_data['values'],
                            marker=marker,
                            s=size,
                            alpha=alpha,
                            label=f"{protein} {treatment} Rep{fit_data['replicate_num']} points"
                        )

                        temp_range = np.linspace(
                            min(fit_data['temperatures']),
                            max(fit_data['temperatures']),
                            100
                        )

                        # Plot fitted line
                        ax.plot(
                            temp_range,
                            sigmoid(temp_range, *fit_data['popt']),
                            '--',
                            alpha=0.7,
                            label=f"{protein} {treatment} Rep{fit_data['replicate_num']} fitted (R²={fit_data['r_squared']:.2f})"
                        )

                        # Calculate melting point
                        melt_pt_temp = fit_data['popt'][1]
                        melt_pt_val = sigmoid(melt_pt_temp, *fit_data['popt'])

                        # Introduce a small horizontal offset based on replicate index
                        x_offset = (i - (len(replicate_fits) - 1) / 2.0) * 0.2

                        # Plot melting point marker with offset
                        ax.scatter(
                            melt_pt_temp + x_offset,
                            melt_pt_val,
                            color='red',
                            s=75,
                            marker='^'
                        )

                        # Plot melting point text with offset
                        ax.text(
                            melt_pt_temp + x_offset,
                            melt_pt_val,
                            f"{melt_pt_temp:.2f}",
                            color='red',
                            horizontalalignment='center',
                            verticalalignment='bottom'
                        )

        # Only set labels and title if we have a valid figure
        if fig is not None and ax is not None:
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Intensity')
            ax.set_title(f'Fitted Curves for {protein} (All Replicates)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            return protein, fig, summary_data
            
    except Exception as e:
        print(f"Unexpected error processing protein {protein}: {str(e)}")
    
    # Return consistent tuple if anything fails
    return None, None, None
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

def fit_and_plot_replicates(replicate_data, selected_temp, normalize_data, r2_threshold):
    """
    Fit curves and create plots for all replicates with progress bar.
    
    Args:
        replicate_data: Dictionary containing protein replicate data
        selected_temp: Temperature for normalization
        normalize_data: Whether to normalize the data
        r2_threshold: R² threshold for curve fitting
        
    Returns:
        tuple: (figures, summary_table)
    """
    # Get unique proteins
    all_proteins = set()
    for treatment in replicate_data.values():
        all_proteins.update(treatment.keys())
    
    # Create iterators for plot styling
    markers = itertools.cycle(['o', 's', '^', 'v'])
    sizes = itertools.cycle([50, 75, 100])
    alphas = itertools.cycle([1.0, 0.8, 0.6])
    positions = itertools.cycle(['left', 'right'])
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_proteins = len(all_proteins)
    
    # Create process pool
    pool = mp.Pool(processes=mp.cpu_count())
    
    # Prepare arguments for parallel processing
    process_args = [
        (protein, replicate_data, iter(markers), iter(sizes), iter(alphas), iter(positions), 
         selected_temp, normalize_data, r2_threshold) 
        for protein in all_proteins
    ]
    
    # Create an iterator for the results
    results_iter = pool.imap(process_protein_replicates, process_args)
    
    # Process results with progress bar
    figures = {}
    all_summary_data = []
    
    for i, result in enumerate(results_iter):
        # Update progress
        progress = (i + 1) / total_proteins
        progress_bar.progress(progress)
        status_text.text(f"Processing protein {i+1} of {total_proteins}")
        
        # Handle result
        if result is not None:
            protein, fig, summary_data = result
            if protein is not None and summary_data:
                figures[protein] = fig
                all_summary_data.extend(summary_data)
    
    # Clean up
    pool.close()
    pool.join()
    progress_bar.empty()
    status_text.empty()
    
    # Create summary table
    if all_summary_data:
        summary_table = pd.DataFrame(all_summary_data)
        # Convert data types
        summary_table['protein'] = summary_table['protein'].astype(str)
        summary_table['treatment'] = summary_table['treatment'].astype(str)
        summary_table['replicate'] = summary_table['replicate'].astype(int)
        summary_table['melting_point'] = summary_table['melting_point'].astype(float)
        summary_table['residuals'] = summary_table['residuals'].astype(str)
    else:
        summary_table = pd.DataFrame(
            columns=['protein', 'treatment', 'replicate', 'melting_point', 'R²', 'residuals']
        )
    
    return figures, summary_table# Save all figures as SVGs and create a zip file
def save_as_svg(figures, dataframe):
    memory_files = []
    for protein, fig in figures.items():
        svg_io = io.BytesIO()
        fig.savefig(svg_io, format='svg', bbox_inches='tight')
        svg_io.seek(0)
        plt.close(fig)
        memory_files.append((f"{protein}.svg", svg_io.getvalue()))

    csv_io = io.BytesIO()
    dataframe.to_csv(csv_io, index=False)
    csv_io.seek(0)
    memory_files.append(("summary.csv", csv_io.getvalue()))

    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, mode='w') as zip_file:
        for file_name, file_content in memory_files:
            zip_file.writestr(file_name, file_content)

    return zip_io.getvalue()

def calculate_bin_width(data):
    data = data.dropna()  # Remove NaN values
    q25, q75 = np.percentile(data, [25, 75])  # Calculate the 25th and 75th percentiles
    iqr = q75 - q25  # Interquartile range
    bin_width = 2 * iqr * len(data) ** (-1/3)  # Freedman-Diaconis rule
    bin_width = max(bin_width, 1e-3)  # Ensure bin width is not too small
    bins = int(np.ceil((data.max() - data.min()) / bin_width))  # Number of bins
    return max(bins, 10)  # Ensure at least 10 bins

def perform_transformations_and_shapiro_test(averaged_table, transformations_to_apply):
    """
    Perform Shapiro-Wilk test on different transformations of the melting point differences.
    
    Args:
        averaged_table (pd.DataFrame): Table containing averaged data
        transformations_to_apply (list): List of transformations to apply
    """
    # Pivot the averaged table for protein melting point differences (ΔTm)
    df_pivot = averaged_table.pivot(index='protein', columns='treatment', values='melting_point')
    df_pivot['ΔTm'] = df_pivot.iloc[:, 0] - df_pivot.iloc[:, 1]
    delta_tm = df_pivot['ΔTm'].dropna()

    # Dictionary to store different transformations
    transformations = {'Original ΔTm': delta_tm}
    
    # Apply selected transformations
    if 'Log' in transformations_to_apply:
        delta_tm_pos = delta_tm - delta_tm.min() + 1  # Ensure positive values
        transformations['Log(ΔTm + shift)'] = np.log(delta_tm_pos)

    if 'Square Root' in transformations_to_apply:
        delta_tm_pos = delta_tm - delta_tm.min()  # Shift to non-negative
        transformations['Square Root of ΔTm'] = np.sqrt(delta_tm_pos)

    if 'Box-Cox' in transformations_to_apply:
        delta_tm_pos = delta_tm - delta_tm.min() + 1  # Ensure positive values
        transformations['Box-Cox ΔTm'] = pd.Series(boxcox(delta_tm_pos)[0], index=delta_tm.index)

    if 'Yeo-Johnson' in transformations_to_apply:
        transformations['Yeo-Johnson ΔTm'] = pd.Series(yeojohnson(delta_tm)[0], index=delta_tm.index)

    # Perform tests and create visualizations
    for name, transformed_data in transformations.items():
        st.subheader(f"Analysis of {name}")
        
        # Perform Shapiro-Wilk test
        stat, p_value = shapiro(transformed_data)
        
        st.write("Shapiro-Wilk test results:")
        st.write(f"- Test statistic: {stat:.4f}")
        st.write(f"- p-value: {p_value:.4f}")
        st.write(f"- {'Normally distributed' if p_value > 0.05 else 'Not normally distributed'}")
        
        # Calculate optimal number of bins using Freedman-Diaconis rule
        bins = calculate_bin_width(transformed_data)
        
        # Create distribution plot
        fig = go.Figure()
        
        # Add histogram with calculated number of bins
        hist = go.Histogram(
            x=transformed_data,
            nbinsx=bins,
            name='Histogram',
            marker=dict(color='lightblue', line=dict(color='black', width=1)),
            opacity=0.7
        )
        fig.add_trace(hist)
        
        # Add density curve
        kde = gaussian_kde(transformed_data)
        kde_x = np.linspace(transformed_data.min(), transformed_data.max(), 200)
        kde_y = kde(kde_x)
        
        # Scale density curve to match histogram counts
        bin_width = (transformed_data.max() - transformed_data.min()) / bins
        scaled_kde_y = kde_y * len(transformed_data) * bin_width
        
        density_curve = go.Scatter(
            x=kde_x,
            y=scaled_kde_y,
            mode='lines',
            name='Density Curve',
            line=dict(color='darkblue', width=2)
        )
        fig.add_trace(density_curve)
        
        fig.update_layout(
            title=f"Distribution of {name}",
            xaxis_title=name,
            yaxis_title="Frequency",
            template='plotly_white',
            barmode='overlay',
            bargap=0.1,
            showlegend=True,
            legend=dict(x=0.7, y=0.95)
        )
        
        st.plotly_chart(fig)

def plot_melting_point_distribution(summary_table):
    st.subheader("Distribution of Melting Points")

    data = summary_table['melting_point']
    data = data.dropna()
    data = data[(data >= 0) & (data <= 100)]  # Filter data within 0-100°C

    if data.empty:
        st.error("No melting point data available for plotting.")
        return

    bins = calculate_bin_width(data)

    # Create histogram with counts (frequency)
    hist = go.Histogram(
        x=data,
        nbinsx=bins,
        marker=dict(color='lightblue', line=dict(color='black', width=1)),
        opacity=0.7,
        name='Histogram'
    )

    # Compute density curve
    kde_x = np.linspace(data.min(), data.max(), 200)
    kde = gaussian_kde(data)
    kde_y = kde(kde_x)

    # Scale density curve to match histogram counts
    bin_width = (data.max() - data.min()) / bins
    scaled_kde_y = kde_y * len(data) * bin_width

    density_curve = go.Scatter(
        x=kde_x,
        y=scaled_kde_y,
        mode='lines',
        line=dict(color='darkblue', width=2),
        name='Density Curve'
    )

    # Combine the histogram and density curve
    fig = go.Figure(data=[hist, density_curve])

    fig.update_layout(
        title="Overall Distribution of Melting Points",
        xaxis_title="Melting Point (°C)",
        yaxis_title="Frequency",
        template='plotly_white',
        bargap=0.1,
        legend=dict(x=0.7, y=0.95)
    )

    st.plotly_chart(fig)

def compare_melting_points_violin(averaged_table):
    """
    Create violin plots comparing melting points between treatments with jittered points.
    """
    st.subheader("Comparison of Melting Points Between Treatments (Interactive Violin Plot)")

    # Ensure 'melting point' and 'treatment' columns exist
    if 'melting_point' not in averaged_table.columns or 'treatment' not in averaged_table.columns:
        st.error("The summary table must contain 'melting_point' and 'treatment' columns.")
        return

    # Prepare data
    data = averaged_table[['melting_point', 'treatment']].dropna()
    data = data[(data['melting_point'] >= 0) & (data['melting_point'] <= 100)]

    # Assign colors to treatments
    treatments = data['treatment'].unique()
    colors = px.colors.qualitative.Plotly
    color_map = dict(zip(treatments, colors))
    treatment_to_num = {treatment: idx for idx, treatment in enumerate(treatments)}

    # Create violin plots
    fig = go.Figure()

    for treatment in treatments:
        treatment_data = data[data['treatment'] == treatment]['melting_point']
        violin_color = color_map[treatment]
        treatment_num = treatment_to_num[treatment]

        # Add violin plot for each treatment
        fig.add_trace(go.Violin(
            y=treatment_data,
            x=[treatment_num] * len(treatment_data),
            name=f"{treatment} Violin",
            box_visible=True,
            meanline_visible=True,
            opacity=0.6,
            hoverinfo='y',
            width=0.6,
            points=False,  
            line_color='black',
            fillcolor=violin_color,
            showlegend=True
        ))

        # Add jittered points
        jitter_strength = 0.05  
        jittered_x = treatment_num + np.random.uniform(-jitter_strength, jitter_strength, size=len(treatment_data))

        fig.add_trace(go.Scatter(
            y=treatment_data,
            x=jittered_x,
            mode='markers',
            name=f"{treatment} Points",
            marker=dict(color='black', size=6, opacity=0.8),
            hoverinfo='y',
            showlegend=True
        ))

    fig.update_layout(
        title="Melting Point Comparison Across Treatments",
        xaxis_title="Treatment",
        yaxis_title="Melting Point (°C)",
        template='plotly_white',
        violingap=0.5,
        violingroupgap=0,
        violinmode='overlay',
        width=800,
        height=600,
        xaxis=dict(
            tickmode='array',
            tickvals=list(treatment_to_num.values()),
            ticktext=list(treatment_to_num.keys())
        )
    )

    st.plotly_chart(fig)

def perform_protein_statistical_tests(replicate_table, treatment_1, treatment_2):
    """
    Perform Mann-Whitney U test on melting points between treatments for each protein,
    with Benjamini-Hochberg FDR correction.
    
    Args:
        replicate_table (pd.DataFrame): Table containing replicate-level melting points
        treatment_1 (str): Name of first treatment (control)
        treatment_2 (str): Name of second treatment
    """
    
    st.subheader("Statistical Analysis of Protein Melting Point Shifts")
    st.write(f"Comparing {treatment_2} vs {treatment_1} (control)")
    
    # Get list of all proteins
    proteins = replicate_table['protein'].unique()
    
    # Store results
    results = []
    raw_p_values = []  # Store raw p-values for FDR correction
    
    # Process each protein
    for protein in proteins:
        # Get data for this protein
        protein_data = replicate_table[replicate_table['protein'] == protein]
        control_data = protein_data[protein_data['treatment'] == treatment_1]['melting_point']
        treatment_data = protein_data[protein_data['treatment'] == treatment_2]['melting_point']
        
        # Skip if insufficient data
        if len(control_data) < 2 or len(treatment_data) < 2:
            continue
            
        try:
            # Perform Mann-Whitney U test
            statistic, p_value = mannwhitneyu(treatment_data, control_data, alternative='two-sided')
            
            # Calculate effect size (r = Z / sqrt(N))
            n1, n2 = len(treatment_data), len(control_data)
            z_score = statistic - (n1 * n2 / 2)
            z_score = z_score / np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
            effect_size = abs(z_score) / np.sqrt(n1 + n2)
            
            # Calculate median difference
            median_diff = treatment_data.median() - control_data.median()
            
            results.append({
                'Protein': protein,
                'Control Tm': f"{control_data.median():.2f} ± {control_data.std():.2f}",
                'Treatment Tm': f"{treatment_data.median():.2f} ± {treatment_data.std():.2f}",
                'ΔTm': f"{median_diff:.2f}",
                'P-value': p_value,
                'Effect Size': effect_size,
                'n_control': len(control_data),
                'n_treatment': len(treatment_data),
                '_sort_tm': abs(median_diff),  # Hidden column for sorting
                '_sort_p': p_value,  # Hidden column for sorting
            })
            raw_p_values.append(p_value)
            
        except Exception as e:
            st.warning(f"Could not analyze protein {protein}: {str(e)}")
    
    if not results:
        st.warning("No proteins had sufficient replicates for statistical analysis.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate Benjamini-Hochberg FDR
    sorted_p_idx = np.argsort(raw_p_values)
    p_values = np.array(raw_p_values)[sorted_p_idx]
    n_tests = len(p_values)
    
    # Calculate FDR
    fdr_values = np.zeros_like(p_values)
    for i, p_value in enumerate(p_values):
        fdr_values[i] = p_value * n_tests / (i + 1)
    
    # Correct for monotonicity
    for i in range(len(fdr_values)-2, -1, -1):
        fdr_values[i] = min(fdr_values[i], fdr_values[i+1])
    
    # Return to original order
    inverse_sorted_p_idx = np.argsort(sorted_p_idx)
    fdr_values = fdr_values[inverse_sorted_p_idx]
    
    # Add FDR values to results
    results_df['FDR'] = fdr_values
    results_df['Significant'] = results_df['FDR'] < 0.05
    
    # Sort by absolute ΔTm and significance
    results_df = results_df.sort_values(
        by=['Significant', '_sort_tm', '_sort_p'],
        ascending=[False, False, True]
    )
    
    # Add significance stars based on FDR
    results_df['Significance'] = results_df['FDR'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    )
    
    # Remove hidden sorting columns
    display_df = results_df.drop(['_sort_tm', '_sort_p'], axis=1)
    
    # Display summary of significant changes
    n_significant = results_df['Significant'].sum()
    st.write(f"Found {n_significant} proteins with significant changes (FDR < 0.05) out of {len(results_df)} tested proteins.")
    
    # Create interactive table
    st.write("### Detailed Results")
    st.write("Click column headers to sort. Significance levels: * FDR<0.05, ** FDR<0.01, *** FDR<0.001, ns: not significant")
    
    # Format p-values and FDR for display
    display_df['P-value'] = display_df['P-value'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}")
    display_df['FDR'] = display_df['FDR'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}")
    display_df['Effect Size'] = display_df['Effect Size'].apply(lambda x: f"{x:.3f}")
    
    # Display table with highlighting
    st.dataframe(
        display_df.style.apply(lambda x: ['background: rgba(144, 238, 144, 0.2)' if x['Significant'] else '' for i in x], axis=1)
    )
    
    # Create volcano plot
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=[float(x.strip()) for x in results_df['ΔTm']],
        y=-np.log10(results_df['FDR']),  # Changed from p-value to FDR
        mode='markers',
        marker=dict(
            color=results_df['Significant'].map({True: 'red', False: 'gray'}),
            size=8,
            opacity=0.7
        ),
        text=results_df['Protein'],  # Hover text
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "ΔTm: %{x:.2f}°C<br>" +
            "-log10(FDR): %{y:.2f}<br>" +  # Updated label
            "<extra></extra>"
        )
    ))
    
    # Add significance threshold line
    fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red", opacity=0.5)
    
    fig.update_layout(
        title="Volcano Plot of Melting Point Changes",
        xaxis_title="ΔTm (°C)",
        yaxis_title="-log10(FDR)",  # Updated label
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig)

def get_species_list():
    """
    Get list of available species from the database.
    
    Returns:
        list: List of species names
    """
    conn = duckdb.connect('multi_proteome_go.duckdb')
    try:
        species = conn.execute("SELECT name FROM species").fetchall()
        species_names = [s[0] for s in species]
        return species_names
    finally:
        conn.close()

def annotate_proteins(protein_data, protein_id_column, selected_species):
    """
    Add GO annotations to a protein dataset.
    
    Args:
        protein_data (pd.DataFrame): DataFrame containing protein IDs
        protein_id_column (str): Name of column containing protein IDs
        selected_species (str): Name of selected species
        
    Returns:
        pd.DataFrame: DataFrame with added GO annotations
    """
    conn = duckdb.connect('multi_proteome_go.duckdb')
    try:
        with st.spinner("Adding GO annotations..."):
            protein_ids = protein_data[protein_id_column].astype(str).tolist()
            annotations = get_go_annotations(conn, protein_ids, selected_species)
            
            # Add annotation columns
            protein_data['GO ID'] = protein_data[protein_id_column].apply(
                lambda pid: ';'.join(annotations.get(pid, {'GO ID': ['NA']})['GO ID'])
            )
            protein_data['Function'] = protein_data[protein_id_column].apply(
                lambda pid: ';'.join(annotations.get(pid, {'Function': ['NA']})['Function'])
            )
            protein_data['Protein Name'] = protein_data[protein_id_column].apply(
                lambda pid: annotations.get(pid, {'Protein Name': 'NA'})['Protein Name']
            )
            protein_data['Link'] = protein_data[protein_id_column].apply(
                lambda pid: annotations.get(pid, {'Link': 'NA'})['Link']
            )
            
            return protein_data
    finally:
        conn.close()

def go_annotation():
    """
    Standalone GO annotation interface.
    """
    st.title("GO Annotation Tool")

    protein_csv = st.file_uploader("Upload Protein CSV file", type=['csv'])

    if protein_csv is not None:
        try:
            protein_data = pd.read_csv(protein_csv)
            st.write("Preview of uploaded data:")
            st.dataframe(protein_data.head())

            protein_id_column = st.selectbox("Select the column containing Protein IDs", protein_data.columns)

            if protein_id_column:
                species_names = get_species_list()
                selected_species = st.selectbox("Select species for GO annotation", species_names)

                if st.button("Start Annotation"):
                    annotated_data = annotate_proteins(protein_data, protein_id_column, selected_species)
                    
                    st.subheader("Updated CSV Data with GO Annotations")
                    st.dataframe(annotated_data)

                    # Offer download of annotated data
                    csv = annotated_data.to_csv(index=False)
                    st.download_button(
                        label="Download Annotated CSV",
                        data=csv,
                        file_name=f"annotated_proteins_{selected_species.replace(' ', '_')}.csv",
                        mime='text/csv',
                    )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def save_results_to_zip(figures, replicate_table, averaged_table):
    zip_buffer = io.BytesIO()
    
    # Create memory files in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        svg_files = pool.starmap(
            save_figure_to_svg,
            [(name, fig) for name, fig in figures.items()]
        )
    
    with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_STORED) as zip_file:
        # Add SVG files
        for name, svg_data in svg_files:
            zip_file.writestr(f"figures/{name}.svg", svg_data)
            
        # Add CSV files without compression
        summary_csv = io.StringIO()
        replicate_table.to_csv(summary_csv, index=False)
        zip_file.writestr("data/summary_table.csv", summary_csv.getvalue())
        
        averaged_csv = io.StringIO()
        averaged_table.to_csv(averaged_csv, index=False)
        zip_file.writestr("data/averaged_summary.csv", averaged_csv.getvalue())
    
    return zip_buffer.getvalue()

def save_figure_to_svg(name, fig):
    svg_io = io.BytesIO()
    fig.savefig(svg_io, format='svg', bbox_inches='tight')
    svg_io.seek(0)
    plt.close(fig)  # Ensure figure is closed immediately
    return name, svg_io.getvalue()

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

def get_go_annotations(conn, protein_ids, species_name):
    query = """
    SELECT p.id AS protein_id, p.name AS protein_name, g.id AS go_id, g.function, p.link
    FROM proteins p
    JOIN protein_go_terms pgt ON p.id = pgt.protein_id
    JOIN go_terms g ON pgt.go_term_id = g.id
    JOIN species s ON p.species_id = s.id
    WHERE p.id IN (SELECT CAST(unnest(?) AS VARCHAR)) AND s.name = ?
    """
    result = conn.execute(query, [protein_ids, species_name]).fetchall()
    
    annotations = {}
    for row in result:
        protein_id, protein_name, go_id, function, link = row
        if protein_id not in annotations:
            annotations[protein_id] = {
                'Protein Name': protein_name,
                'GO ID': [],
                'Function': [],
                'Link': link
            }
        annotations[protein_id]['GO ID'].append(go_id)
        annotations[protein_id]['Function'].append(function)
    
    return annotations

def visualize_go_ids_with_dtm(averaged_table, threshold, treatment_1, treatment_2):
    """
    Visualize GO IDs with significant ΔTm using averaged data.
    """
    # Create pivot table from averaged data
    pivot_table = averaged_table.pivot(index='protein', columns='treatment', values='melting_point')

    if treatment_1 in pivot_table.columns and treatment_2 in pivot_table.columns:
        pivot_table['ΔTm'] = pivot_table[treatment_1] - pivot_table[treatment_2]
    else:
        st.error(f"The selected treatments '{treatment_1}' and '{treatment_2}' are missing in the summary table.")
        return

    filtered_table = pivot_table[pivot_table['ΔTm'].abs() > threshold].reset_index()

    if filtered_table.empty:
        st.warning(f"No proteins found with ΔTm greater than {threshold}°C.")
        return

    # Get unique GO IDs and Functions for each protein
    go_data = averaged_table[['protein', 'GO ID', 'Function']].drop_duplicates(subset=['protein'])
    filtered_table = filtered_table.merge(go_data, on='protein', how='inner')

    filtered_table = filtered_table[filtered_table['GO ID'].notna() & (filtered_table['GO ID'] != 'NA')]

    if filtered_table.empty:
        st.warning("No proteins with valid GO IDs after filtering.")
        return

    # Process GO data
    filtered_table['GO ID'] = filtered_table['GO ID'].str.split(';')
    filtered_table['Function'] = filtered_table['Function'].str.split(';')
    exploded_table = filtered_table.explode(['GO ID', 'Function'])

    function_counts = exploded_table.groupby('Function').size().reset_index(name='counts')
    function_go_ids = exploded_table.groupby('Function')['GO ID'].apply(lambda x: '; '.join(set(x))).reset_index()
    function_data = function_counts.merge(function_go_ids, on='Function')
    
    # Sort and limit to top 20
    function_data = function_data.sort_values(by='counts', ascending=False).head(20)
    
    # Create visualization
    max_label_length = 30
    function_data['Wrapped Function'] = function_data['Function'].apply(
        lambda x: '<br>'.join(textwrap.wrap(x, width=max_label_length))
    )

    fig = go.Figure(data=[
        go.Bar(
            x=function_data['Wrapped Function'],
            y=function_data['counts'],
            hovertemplate='<b>Function:</b> %{customdata[0]}<br><b>Count:</b> %{y}<br><b>GO IDs:</b> %{customdata[1]}<extra></extra>',
            customdata=np.stack((function_data['Function'], function_data['GO ID']), axis=-1),
            marker=dict(color='skyblue'),
        )
    ])

    fig.update_layout(
        title=f"Top 20 Functions with ΔTm > {threshold}°C (Comparing '{treatment_1}' vs '{treatment_2}')",
        xaxis_title="Function",
        yaxis_title="Frequency",
        template='plotly_white',
        xaxis_tickangle=-90,
        xaxis=dict(
            automargin=True,
            tickmode='linear',
            tickfont=dict(size=10),
        ),
        margin=dict(b=150),
    )

    st.plotly_chart(fig)

def session_init():
    if 'tsv_data' not in st.session_state:
        st.session_state.tsv_data = None
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = None
    if 'edit_mode_tsv' not in st.session_state:
        st.session_state.edit_mode_tsv = False
    if 'edit_mode_csv' not in st.session_state:
        st.session_state.edit_mode_csv = False
    if 'normalize_data' not in st.session_state:
        st.session_state.normalize_data = True
    if 'selected_temp' not in st.session_state:
        st.session_state.selected_temp = None
    if 'perform_shapiro' not in st.session_state:
        st.session_state.perform_shapiro = False
    if 'transformations_to_apply' not in st.session_state:
        st.session_state.transformations_to_apply = []
    if 'visualize_go_ids' not in st.session_state:
        st.session_state.visualize_go_ids = False
    if 'threshold' not in st.session_state: 
        st.session_state.threshold = 4.0

def validate_inputs(uploaded_tsv, uploaded_csv):
    """
    Validate user uploaded files and their content.
    """
    if uploaded_tsv is None or uploaded_csv is None:
        st.warning("Please upload both TSV and CSV files before loading.")
        return False
        
    if not uploaded_tsv.name.endswith('.tsv'):
        st.error("First file must be a TSV file.")
        return False
        
    if not uploaded_csv.name.endswith('.csv'):
        st.error("Second file must be a CSV file.")
        return False
        
    return True

def clear_session_data():
    """
    Clear all session state data.
    """
    st.session_state.tsv_data = None
    st.session_state.csv_data = None
    st.session_state.edit_mode_tsv = False
    st.session_state.edit_mode_csv = False

def setup_data_loading_interface():
    """
    Create and handle the data loading interface elements.
    """
    uploaded_tsv = st.file_uploader("Upload TSV fragpipe output file", type=['tsv'])
    uploaded_csv = st.file_uploader("Upload CSV metadata file", type=['csv'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button('Load uploaded data'):
            if validate_inputs(uploaded_tsv, uploaded_csv):
                try:
                    st.session_state.tsv_data = read_tsv_file(uploaded_tsv)
                    st.session_state.csv_data = read_csv_file(uploaded_csv)
                    st.success("Uploaded data loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading files: {str(e)}")
    
    with col2:
        handle_example_data()
    
    with col3:
        if st.button('Clear loaded data'):
            clear_session_data()
            st.success("All loaded data has been cleared!")
            
    return uploaded_tsv, uploaded_csv

def handle_example_data():
    """
    Handle loading of example data.
    """
    sample_help = "By pressing this button, sample experimental data will be loaded for demonstration purposes"
    if st.button('Load example data', help=sample_help):
        try:
            data_path = os.path.dirname(os.path.abspath(__file__))
            tsv_path = os.path.join(data_path, "sample_data.tsv")
            csv_path = os.path.join(data_path, "sample_metadata.csv")
            
            if not os.path.exists(tsv_path) or not os.path.exists(csv_path):
                st.error("Sample data files not found!")
                return
                
            st.session_state.tsv_data = read_tsv_file(tsv_path)
            st.session_state.csv_data = read_csv_file(csv_path)
            st.success("Sample data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")

def display_data_tables():
    """
    Display and handle the TSV and CSV data tables.
    """
    if st.session_state.tsv_data is not None:
        with st.expander("TSV Data", expanded=True):
            handle_tsv_display()

    if st.session_state.csv_data is not None:
        with st.expander("CSV Metadata", expanded=True):
            handle_csv_display()

def handle_tsv_display():
    """
    Handle the TSV data display and editing interface.
    """
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

def handle_csv_display():
    """
    Handle the CSV data display and editing interface.
    """
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

def setup_analysis_parameters(metadata):
    """
    Set up and handle analysis parameters interface.
    """
    normalize_data = handle_normalization(metadata)
    return normalize_data

def handle_normalization(metadata):
    """
    Handle normalization settings interface.
    """
    normalize_data = st.checkbox("Normalize", value=st.session_state.get('normalize_data', True))
    
    if normalize_data:
        unique_temperatures = sorted(set(metadata['Temperature']))
        selected_temp = st.selectbox(
            "Select the temperature to which data should be normalized:",
            options=unique_temperatures,
            index=unique_temperatures.index(st.session_state.get('selected_temp', unique_temperatures[0])) 
            if st.session_state.get('selected_temp') in unique_temperatures else 0
        )
        st.session_state.normalize_data = normalize_data
        st.session_state.selected_temp = selected_temp
    else:
        st.session_state.selected_temp = None

    if st.session_state.selected_temp is None and normalize_data:
        st.warning("Please select a temperature for normalization.")
        
    return normalize_data

def setup_go_annotation():
    """
    Handle GO annotation setup interface.
    """
    include_go_annotation = st.checkbox(
        "Include GO annotation",
        value=False,
        help="Adds Gene Ontology (GO) terms and functions to proteins in the analysis. "
             "GO annotations help understand the biological roles, molecular functions, "
             "and cellular locations of the proteins being studied."
    )

    selected_species = None
    if include_go_annotation:
        conn = duckdb.connect('multi_proteome_go.duckdb')
        species = conn.execute("SELECT name FROM species").fetchall()
        species_names = [s[0] for s in species]
        selected_species = st.selectbox("Select species for GO annotation", species_names)
        conn.close()

    return include_go_annotation, selected_species

def setup_visualization_options(treatments, include_go_annotation):
    """
    Create and handle visualization option interface.
    """
    with st.expander("Data Visualization Options"):
        show_distribution = st.checkbox(
            "Show Distribution of Melting Points",
            value=False,
            help="Displays a histogram and density plot showing the overall distribution of melting points across all proteins. "
                 "This helps visualize the spread and central tendency of your melting point data."
        )
        
        show_violin_plot = st.checkbox(
            "Compare Melting Points Between Treatments (Violin Plot)",
            value=False,
            help="Creates an interactive violin plot comparing melting point distributions between treatments. "
                 "The plot includes individual data points and shows the shape, median, and quartiles of the distribution for each treatment."
        )
        
        show_statistics = st.checkbox(
            "Mann-Whitney and Benjamini-Hochberg",
            value=False,
            help="Performs Mann-Whitney U tests to compare melting points between treatments for each protein, "
                 "with Benjamini-Hochberg correction for multiple testing. Generates a volcano plot and interactive table "
                 "showing significant changes in protein stability between conditions."
        )
        
        treatment_1, treatment_2 = None, None
        if show_statistics or st.session_state.visualize_go_ids:
            treatment_1, treatment_2 = select_treatments(treatments)
            
        visualize_go_ids = handle_go_visualization(include_go_annotation)

    return show_distribution, show_violin_plot, show_statistics, visualize_go_ids, treatment_1, treatment_2

def handle_go_visualization(include_go_annotation):
    """
    Handle GO visualization interface.
    """
    visualize_go_ids = False
    if include_go_annotation:
        visualize_go_ids = st.checkbox(
            "Visualize GO IDs with ΔTm > X°C",
            value=False,
            help="Creates a bar chart showing the most frequent GO terms for proteins with significant "
                 "melting point changes. Helps identify which biological processes or molecular functions "
                 "are most affected by the treatment."
        )
        if visualize_go_ids:
            st.session_state['threshold'] = st.number_input(
                "ΔTm threshold for GO ID visualization",
                value=4.0,
                step=0.1
            )
    st.session_state.visualize_go_ids = visualize_go_ids
    return visualize_go_ids

def setup_shapiro_wilk_test():
    """
    Handle Shapiro-Wilk test setup interface.
    """
    with st.expander("Shapiro-Wilk Normality Test"):
        perform_shapiro = st.checkbox(
            "Perform Shapiro-Wilk Test",
            value=False,
            help="Tests whether the ΔTm (melting point differences) follow a normal distribution. "
                 "The test is performed on the original data and selected transformations to identify "
                 "which form of the data best approximates normality."
        )

        transformations = handle_transformation_options(perform_shapiro)

        if st.button("Apply Shapiro-Wilk Test Settings"):
            st.session_state.perform_shapiro = perform_shapiro
            if perform_shapiro:
                st.session_state.transformations_to_apply = transformations
            st.success("Shapiro-Wilk Test settings applied successfully!")
            
    return perform_shapiro, transformations

def handle_transformation_options(perform_shapiro):
    """
    Handle transformation options interface.
    """
    st.subheader("Normality Test Selection")
    transformations = []
    
    if st.checkbox("Log Transformation", value=False, disabled=not perform_shapiro,
                  help="Applies natural logarithm to the data. Useful for right-skewed distributions "
                       "and when the data spans multiple orders of magnitude."):
        transformations.append("Log")
        
    if st.checkbox("Square Root Transformation", value=False, disabled=not perform_shapiro,
                  help="Takes the square root of the data. A milder transformation than log, "
                       "useful for right-skewed data and count data."):
        transformations.append("Square Root")
        
    if st.checkbox("Box-Cox Transformation", value=False, disabled=not perform_shapiro,
                  help="A family of power transformations that includes log and square root as special cases. "
                       "Automatically finds the optimal power parameter to make data as normal as possible. "
                       "Only works with positive values."):
        transformations.append("Box-Cox")
        
    if st.checkbox("Yeo-Johnson Transformation", value=False, disabled=not perform_shapiro,
                  help="Similar to Box-Cox but can handle negative values. A more flexible transformation "
                       "that works well with data containing zeros or negative numbers."):
        transformations.append("Yeo-Johnson")
        
    return transformations

def handle_analysis_results(filtered_data, filtered_data_imputed, replicate_data, figures, 
                          summary_table, show_statistics, show_distribution, show_violin_plot,
                          perform_shapiro, include_go_annotation, visualize_go_ids,
                          treatment_1, treatment_2, averaged_figures=None,
                          averaged_summary_table=None):
    """
    Process and display analysis results.
    """
    st.subheader("Analysis Results")
    st.write(f"Number of rows after filtering: {len(filtered_data)}")
    
    # Generate replicate and averaged tables
    replicate_table, averaged_table = process_summary_tables(summary_table)
    
    # Display visualizations based on user selections
    if show_statistics and treatment_1 and treatment_2:
        perform_protein_statistical_tests(replicate_table, treatment_1, treatment_2)

    if show_distribution:
        plot_melting_point_distribution(summary_table)

    if show_violin_plot:
        compare_melting_points_violin(averaged_table)

    if perform_shapiro:
        perform_transformations_and_shapiro_test(averaged_table, st.session_state['transformations_to_apply'])

    if visualize_go_ids and include_go_annotation:
        visualize_go_ids_with_dtm(averaged_table, st.session_state['threshold'], treatment_1, treatment_2)

    # Create and offer download with timing information
    with st.spinner('Preparing files for download...'):
        start_time = time.time()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        zip_filename = f"TPP_analysis_{timestamp}.zip"
        
        # Create zip file containing both replicate and averaged results
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_STORED) as zip_file:
            # Save replicate-level figures
            with mp.Pool(processes=mp.cpu_count()) as pool:
                svg_files = pool.starmap(
                    save_figure_to_svg,
                    [(name, fig) for name, fig in figures.items()]
                )
            
            for name, svg_data in svg_files:
                zip_file.writestr(f"figures/replicates/{name}.svg", svg_data)

            # Save averaged figures if provided
            if averaged_figures is not None:
                with mp.Pool(processes=mp.cpu_count()) as pool:
                    avg_svg_files = pool.starmap(
                        save_figure_to_svg,
                        [(f"{name}_averaged", fig) for name, fig in averaged_figures.items()]
                    )
                
                for name, svg_data in avg_svg_files:
                    zip_file.writestr(f"figures/averaged/{name}.svg", svg_data)

            # Save summary tables
            summary_csv = io.StringIO()
            replicate_table.to_csv(summary_csv, index=False)
            zip_file.writestr("data/replicate_summary.csv", summary_csv.getvalue())

            if averaged_summary_table is not None:
                averaged_csv = io.StringIO()
                averaged_summary_table.to_csv(averaged_csv, index=False)
                zip_file.writestr("data/averaged_summary.csv", averaged_csv.getvalue())
        
        end_time = time.time()
        st.write(f"Time taken to save files: {end_time - start_time:.2f} seconds")
        
        st.download_button(
            label="Download Analysis Results",
            data=zip_buffer.getvalue(),
            file_name=zip_filename,
            mime="application/zip"
        )

    # Display summary tables
    st.subheader("Summary Tables")
    tab1, tab2 = st.tabs(["Replicate Data", "Averaged Data"])

    with tab1:
        st.dataframe(replicate_table)

    with tab2:
        if averaged_summary_table is not None:
            st.dataframe(averaged_summary_table)
        else:
            st.dataframe(averaged_table)

def add_go_annotations(summary_table, selected_species):
    """
    Add GO annotations to the summary table.
    
    Args:
        summary_table (pd.DataFrame): DataFrame containing the protein analysis results
        selected_species (str): Name of the selected species for GO annotation
        
    Returns:
        pd.DataFrame: Summary table with added GO annotations
    """
    conn = duckdb.connect('multi_proteome_go.duckdb')
    
    with st.spinner("Adding GO annotations..."):
        protein_ids = summary_table['protein'].tolist()
        annotations = get_go_annotations(conn, protein_ids, selected_species)

        # Add new columns for annotations
        for index, row in summary_table.iterrows():
            protein_id = row['protein']
            annotation = annotations.get(protein_id, {
                'Protein Name': 'NA',
                'GO ID': ['NA'],
                'Function': ['NA'],
                'Link': 'NA'
            })
            summary_table.at[index, 'Protein Name'] = annotation['Protein Name']
            summary_table.at[index, 'GO ID'] = ';'.join(annotation['GO ID'])
            summary_table.at[index, 'Function'] = ';'.join(annotation['Function'])
            summary_table.at[index, 'Link'] = annotation['Link']

        st.write("GO annotations and protein names added to the summary table.")

    conn.close()

    # Reorder columns to keep consistent format
    desired_order = ['protein', 'Protein Name', 'treatment', 'melting_point', 
                    'R²', 'residuals', 'GO ID', 'Function', 'Link']
    summary_table = summary_table[desired_order]
    
    return summary_table

def display_summary_tables(summary_table):
    """
    Display the summary tables in tabs.
    """
    st.subheader("Summary Tables")
    tab1, tab2 = st.tabs(["Original Data", "Averaged Data"])

    with tab1:
        st.dataframe(summary_table)

    with tab2:
        replicate_table, averaged_table = process_summary_tables(summary_table)
        st.write("Averaged data showing mean values and standard deviations:")
        st.dataframe(averaged_table)

def select_treatments(treatments):
    """
    Handle treatment selection interface.
    
    Args:
        treatments: List of available treatments
    """
    if 'treatment_1' not in st.session_state:
        st.session_state['treatment_1'] = treatments[0]
    if 'treatment_2' not in st.session_state:
        st.session_state['treatment_2'] = treatments[1]

    treatment_1 = st.selectbox(
        "Select first treatment (control):",
        treatments,
        key='treatment_1'
    )
    treatment_2 = st.selectbox(
        "Select second treatment:",
        treatments,
        key='treatment_2'
    )
    
    return treatment_1, treatment_2

def create_results_zip(figures, summary_table, averaged_figures=None, averaged_summary_table=None):
    """
    Create a zip file containing analysis results including replicate and averaged figures.
    
    Args:
        figures (dict): Dictionary of replicate-level figures keyed by protein.
        summary_table (pd.DataFrame): Summary table of replicate-level results.
        averaged_figures (dict, optional): Dictionary of averaged figures keyed by (protein, treatment).
        averaged_summary_table (pd.DataFrame, optional): Summary table of averaged data.
        
    Returns:
        io.BytesIO: In-memory zip file containing all results.
    """
    # Process summary tables to get replicate and averaged versions
    replicate_table, default_averaged_table = process_summary_tables(summary_table)

    # If an averaged summary table is not provided, default to the one from process_summary_tables
    if averaged_summary_table is None:
        averaged_summary_table = default_averaged_table

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_STORED) as zip_file:
        # Save replicate-level figures
        with mp.Pool(processes=mp.cpu_count()) as pool:
            svg_files = pool.starmap(
                save_figure_to_svg,
                [(name, fig) for name, fig in figures.items()]
            )
        
        for name, svg_data in svg_files:
            zip_file.writestr(f"figures/{name}.svg", svg_data)

        # Save averaged figures if provided
        if averaged_figures is not None:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                avg_svg_files = pool.starmap(
                    save_figure_to_svg,
                    [(f"{protein}_{treatment}", fig) for (protein, treatment), fig in averaged_figures.items()]
                )
            
            # Write averaged figures into the figures/average directory
            for name, svg_data in avg_svg_files:
                zip_file.writestr(f"figures/average/{name}.svg", svg_data)

        # Add replicate summary table
        summary_csv = io.StringIO()
        replicate_table.to_csv(summary_csv, index=False)
        zip_file.writestr("data/summary_table.csv", summary_csv.getvalue())

        # Add averaged summary table
        averaged_csv = io.StringIO()
        averaged_summary_table.to_csv(averaged_csv, index=False)
        zip_file.writestr("data/averaged_summary.csv", averaged_csv.getvalue())

    return zip_buffer


def fit_and_plot_averaged_curves(replicate_data, selected_temp=None, normalize_data=True, r2_threshold=0.8, selected_species=None, include_go_annotation=False):
    """
    Fit sigmoidal curves to averaged data with spaced melting point markers.
    """
    figures = {}
    summary_data = []
    all_proteins = set()
    
    for treatment, proteins_dict in replicate_data.items():
        for prot in proteins_dict.keys():
            all_proteins.add(prot)

    total_proteins = len(all_proteins)
    progress_bar = st.progress(0)
    status_text = st.empty()
    st.write("Fitting averaged curves...")
    
    for i, protein in enumerate(all_proteins):
        progress = (i + 1) / total_proteins
        progress_bar.progress(progress)
        status_text.text(f"Processing averaged curves: protein {i+1} of {total_proteins}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        successful_fits = []

        for treatment, proteins_dict in replicate_data.items():
            if protein not in proteins_dict:
                continue
                
            temp_dict = proteins_dict[protein]
            temps = sorted(temp_dict.keys())

            if len(temps) < 4:
                continue

            intensities_list = [np.mean(np.array(temp_dict[t])) for t in temps]
            temperatures = np.array(temps)
            values = np.array(intensities_list)

            # Remove NaNs
            mask = ~np.isnan(values)
            temperatures = temperatures[mask]
            values = values[mask]

            if len(temperatures) < 4:
                continue

            if normalize_data and selected_temp in temperatures:
                norm_idx = np.where(temperatures == selected_temp)[0][0]
                norm_value = values[norm_idx]
                if norm_value != 0:
                    values = values / norm_value

            try:
                valmax = np.max(values)
                med = np.median(temperatures)
                minval = np.min(values)

                popt, _ = curve_fit(sigmoid, temperatures, values, p0=[valmax, med, minval])
                fitted_values = sigmoid(temperatures, *popt)

                ss_res = np.sum((values - fitted_values)**2)
                ss_tot = np.sum((values - np.mean(values))**2)
                r_squared = 1 - (ss_res / ss_tot)

                if r_squared < r2_threshold:
                    continue

                successful_fits.append({
                    'treatment': treatment,
                    'temperatures': temperatures,
                    'values': values,
                    'popt': popt,
                    'r_squared': r_squared,
                    'fitted_values': fitted_values
                })
                
                summary_data.append({
                    'protein': protein,
                    'treatment': treatment,
                    'melting_point': popt[1],
                    'R²': r_squared,
                    'residuals': ','.join(map(str, values - fitted_values))
                })

            except RuntimeError:
                continue

        if successful_fits:
            successful_fits.sort(key=lambda x: x['popt'][1])
            
            for idx, fit in enumerate(successful_fits):
                # Plot measured points and fitted curve
                ax.scatter(
                    fit['temperatures'], 
                    fit['values'],
                    marker='o', 
                    s=50, 
                    alpha=1.0,
                    label=f"{protein} {fit['treatment']} averaged points"
                )

                temp_range = np.linspace(
                    fit['temperatures'].min(), 
                    fit['temperatures'].max(), 
                    100
                )
                ax.plot(
                    temp_range, 
                    sigmoid(temp_range, *fit['popt']),
                    '--', 
                    alpha=0.7,
                    label=f"{protein} {fit['treatment']} averaged fitted (R²={fit['r_squared']:.2f})"
                )

                # Add spaced melting point marker
                melt_pt_temp = fit['popt'][1]
                melt_pt_val = sigmoid(melt_pt_temp, *fit['popt'])
                offset = 0.3 * (idx - (len(successful_fits) - 1) / 2)  # Reduced spacing

                ax.scatter(
                    melt_pt_temp + offset, 
                    melt_pt_val,
                    color='red', 
                    s=75, 
                    marker='^',
                    zorder=5
                )
                ax.text(
                    melt_pt_temp + offset,
                    melt_pt_val,
                    f"{melt_pt_temp:.2f}°C",
                    color='red',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    fontsize=10,
                    zorder=6  
                )

            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel('Normalized Intensity' if normalize_data else 'Intensity')
            ax.set_title(f'Fitted Curves for {protein} (Averaged Data)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            figures[protein] = fig
        else:
            plt.close(fig)

    progress_bar.empty()
    status_text.empty()

    # Create and annotate averaged summary table with same logic as before
    if summary_data:
        averaged_summary_table = pd.DataFrame(summary_data)
        averaged_summary_table['protein'] = averaged_summary_table['protein'].astype(str)
        averaged_summary_table['treatment'] = averaged_summary_table['treatment'].astype(str)
        averaged_summary_table['melting_point'] = averaged_summary_table['melting_point'].astype(float)
        averaged_summary_table['residuals'] = averaged_summary_table['residuals'].astype(str)
        
        if include_go_annotation and selected_species:
            conn = duckdb.connect('multi_proteome_go.duckdb')
            try:
                st.write("Adding GO annotations to averaged summary table...")
                protein_ids = averaged_summary_table['protein'].unique().tolist()
                annotations = get_go_annotations(conn, protein_ids, selected_species)
                
                for col, default in [('GO ID', ['NA']), ('Function', ['NA']), 
                                   ('Protein Name', 'NA'), ('Link', 'NA')]:
                    averaged_summary_table[col] = averaged_summary_table['protein'].apply(
                        lambda pid: ';'.join(annotations.get(pid, {col: default})[col]) 
                        if isinstance(default, list) 
                        else annotations.get(pid, {col: default})[col]
                    )
                
            finally:
                conn.close()
            
            column_order = ['protein', 'Protein Name', 'treatment', 'melting_point', 
                          'R²', 'residuals', 'GO ID', 'Function', 'Link']
            averaged_summary_table = averaged_summary_table[
                [col for col in column_order if col in averaged_summary_table.columns]
            ]
    else:
        averaged_summary_table = pd.DataFrame(
            columns=['protein', 'treatment', 'melting_point', 'R²', 'residuals']
        )

    return figures, averaged_summary_table

def analysis():
    """
    Main analysis function coordinating the entire workflow.
    """
    try:
        st.title("TPP Analysis App")
        
        session_init()

        # Set up data loading interface
        uploaded_tsv, uploaded_csv = setup_data_loading_interface()

        if st.session_state.tsv_data is not None and st.session_state.csv_data is not None:
            st.info("TSV and CSV data are loaded and ready for analysis.")
            
            # Display data tables
            display_data_tables()
            
            # Setup analysis parameters
            st.subheader("Analysis Setup")
            metadata = extract_samples(st.session_state.csv_data)
            
            if metadata is None:
                st.error("Failed to extract samples from metadata. Please check your CSV file.")
                return
                
            normalize_data = setup_analysis_parameters(metadata)
            
            # Setup GO annotation
            include_go_annotation, selected_species = setup_go_annotation()
            
            # Get treatments list
            treatments = list(set(metadata['Treatment']))
            
            # Setup visualization options
            show_distribution, show_violin_plot, show_statistics, visualize_go_ids, treatment_1, treatment_2 = \
                setup_visualization_options(treatments, include_go_annotation)
            
            # Setup Shapiro-Wilk test
            perform_shapiro, transformations = setup_shapiro_wilk_test()
            
            # R² threshold setup
            st.session_state.r2_threshold = st.slider(
                "Set the R² threshold for filtering proteins:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('r2_threshold', 0.8),
                step=0.01,
                help="Minimum R² value required for accepting protein curve fits"
            )

            # Start analysis button
            if st.button("Start Analysis"):
                try:
                    # Filter and process data
                    filtered_data, ceiling_rand = filter_and_lowest_float(
                        st.session_state.tsv_data,
                        metadata['Samples']
                    )
                    
                    filtered_data_imputed = impute_filtered_data(
                        filtered_data.copy(),
                        metadata['Samples'],
                        ceiling_rand
                    )
                    
                    replicate_data = get_replicate_data(
                        st.session_state.csv_data,
                        filtered_data_imputed
                    )
                    
                    # Fit curves and generate figures
                    with st.spinner('Processing data and generating figures...'):
                        start_time = time.time()
                        figures, summary_table = fit_and_plot_replicates(
                            replicate_data,
                            st.session_state.selected_temp,
                            st.session_state.normalize_data,
                            st.session_state.r2_threshold
                        )
                        end_time = time.time()
                        averaged_figures, averaged_summary_table = fit_and_plot_averaged_curves(
                            replicate_data,
                            selected_temp=st.session_state.selected_temp,
                            normalize_data=st.session_state.normalize_data,
                            r2_threshold=st.session_state.r2_threshold,
                            selected_species=selected_species if include_go_annotation else None,
                            include_go_annotation=include_go_annotation
                        )
                        st.write(f"Time taken to generate figures: {end_time - start_time:.2f} seconds | "
                                f"{len(figures)} figures generated")
                    
                    # Add GO annotations if requested
                    if include_go_annotation:
                        summary_table = add_go_annotations(summary_table, selected_species)
                    
                    # Handle and display results
                    handle_analysis_results(
                        filtered_data,
                        filtered_data_imputed,
                        replicate_data,
                        figures,
                        summary_table,
                        show_statistics,
                        show_distribution,
                        show_violin_plot,
                        perform_shapiro,
                        include_go_annotation,
                        visualize_go_ids,
                        treatment_1,
                        treatment_2,
                        averaged_figures=averaged_figures,
                        averaged_summary_table=averaged_summary_table
                    )
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    raise e
                finally:
                    for fig in plt.get_fignums():
                        plt.close(fig)
        else:
            st.warning("Please load both TSV and CSV data to proceed with analysis.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
    finally:
        # Clean up any remaining matplotlib figures
        plt.close('all')

def main():
    st.set_page_config(
        page_title="TPP Solver",
        page_icon="logo_32x32.png",
        layout="wide",
    )
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Main App", 'GO Annotation', "README"])

    if page == "README":
        display_readme()
    elif page == "Main App":
        analysis()
    elif page == "GO Annotation":
        go_annotation()
    #elif page == "proteome scraper"

if __name__ == "__main__":
    main()