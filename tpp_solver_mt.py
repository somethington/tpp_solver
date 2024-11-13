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
from scipy.stats import shapiro, boxcox, yeojohnson, gaussian_kde
import seaborn as sns
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

def process_protein(args):
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
                temperatures = np.array(list(proteins[protein].keys()))
                values = np.array(list(proteins[protein].values()))

                index = np.argsort(temperatures)
                temperatures = temperatures[index]
                values = values[index]

                if normalize_data:
                    norm_indices = np.where(temperatures == selected_temp)[0]
                    if len(norm_indices) == 0:
                        continue 
                    norm_index = norm_indices[0]
                    values = values / values[norm_index]

                valmax = max(values)
                med = np.median(temperatures)
                minval = min(values)

                popt, _ = curve_fit(
                    sigmoid, temperatures, values, p0=[valmax, med, minval]
                )

                # Predicted values using the fitted sigmoid curve
                fitted_values = sigmoid(temperatures, *popt)

                # Residuals and total sum of squares
                ss_res = np.sum((values - fitted_values) ** 2)
                ss_tot = np.sum((values - np.mean(values)) ** 2)

                # R² calculation
                r_squared = 1 - (ss_res / ss_tot)

                # Filter based on R² threshold
                if r_squared < r2_threshold:
                    continue  # Skip this treatment if R² is below threshold

                # Proceed with plotting
                if fig is None and ax is None:
                    fig, ax = plt.subplots(figsize=(10, 6))

                marker = next(markers)
                size = next(sizes)
                alpha = next(alphas)
                curpos = next(positions)

                ax.scatter(
                    temperatures,
                    values,
                    marker=marker,
                    s=size,
                    alpha=alpha,
                    label=f"{protein} {treatment} measured",
                )

                melt_pt = sigmoid(popt[1], *popt)

                temp_range = np.linspace(temperatures.min(), temperatures.max(), 100)
                ax.plot(
                    temp_range,
                    sigmoid(temp_range, *popt),
                    "--",
                    alpha=0.7,
                    label=f"{protein} {treatment} fitted",
                )

                ax.scatter(popt[1], melt_pt, color="red", s=75, marker="^")
                ax.text(
                    popt[1],
                    melt_pt,
                    f"{popt[1]:.2f}",
                    color="red",
                    horizontalalignment=curpos,
                    verticalalignment="bottom",
                )

                ax.text(
                    0.05,
                    0.95 - y_offset,
                    f"R² ({treatment}) = {r_squared:.2f}",
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
                )

                y_offset += 0.07

                summary_data.append(
                    {
                        "protein": protein,
                        "treatment": treatment,
                        "melting point": popt[1],
                        "R²": r_squared,
                        "residuals": ",".join(map(str, (values - fitted_values))),
                    }
                )

        if fig is not None and ax is not None:
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Intensity")
            ax.set_title(f"Fitted Curve for {protein}")
            ax.legend()

            return protein, fig, summary_data
        else:
            return None, None, None

    except RuntimeError as e:
        print(f"Error processing protein {protein}: {str(e)}")
        return None, None, None
    except Exception as e:
        print(f"Unexpected error processing protein {protein}: {str(e)}")
        return None, None, None
    
def fit_and_plot(data_dict, selected_temp, normalize_data, r2_threshold):
    all_proteins = set()
    for treatment in data_dict.values():
        all_proteins.update(treatment.keys())

    markers = itertools.cycle(['o', 's'])
    sizes = itertools.cycle([50, 100])
    alphas = itertools.cycle([1.0, 0.5])
    positions = itertools.cycle(["left", "right"])

    pool = mp.Pool(processes=mp.cpu_count())
    
    results = pool.map(process_protein, 
                       [(protein, data_dict, iter(markers), iter(sizes), iter(alphas), iter(positions), selected_temp, normalize_data, r2_threshold) 
                        for protein in all_proteins])
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

    if all_summary_data:
        summary_table = pd.DataFrame(all_summary_data)
        summary_table['protein'] = summary_table['protein'].astype(str)
        summary_table['treatment'] = summary_table['treatment'].astype(str)
        summary_table['melting point'] = summary_table['melting point'].astype(float)
        summary_table['residuals'] = summary_table['residuals'].astype(str)
    else:
        summary_table = pd.DataFrame(columns=['protein', 'treatment', 'melting point', 'residuals'])

    return figures, summary_table

# Save all figures as SVGs and create a zip file
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

def perform_transformations_and_shapiro_test(summary_table, transformations_to_apply):
    # Pivot the summary table for protein melting point differences (ΔTm)
    df_pivot = summary_table.pivot(index='protein', columns='treatment', values='melting point')
    df_pivot['ΔTm'] = df_pivot.iloc[:, 0] - df_pivot.iloc[:, 1]
    delta_tm = df_pivot['ΔTm'].dropna()  # Drop NaN values

    # Dictionary to store different transformations
    transformations = {'Original ΔTm': delta_tm}

    # Apply Log(ΔTm + 1) transformation, ensuring only non-negative values
    if 'Log' in transformations_to_apply:
        delta_tm = delta_tm  # Ensure non-negative values for log
        if len(delta_tm) > 0:
            transformations['Log(ΔTm + 1)'] = np.log1p(delta_tm)
        else:
            st.warning("No valid non-negative data for Log(ΔTm + 1) transformation.")

    # Apply Square Root transformation, ensuring only non-negative values
    if 'Square Root' in transformations_to_apply:
        delta_tm = delta_tm
        if len(delta_tm) > 0:
            transformations['Square Root of ΔTm'] = np.sqrt(delta_tm)
        else:
            st.warning("No valid non-negative data for Square Root transformation.")

    # Apply Box-Cox transformation, ensuring all positive values
    if 'Box-Cox' in transformations_to_apply:
        if (delta_tm > 0).all():  # Box-Cox requires positive values
            transformations['Box-Cox ΔTm'] = pd.Series(boxcox(delta_tm)[0], index=delta_tm.index)
        else:
            # Apply Box-Cox with a shift to handle non-positive values
            shifted_delta_tm = delta_tm - delta_tm.min() + 1
            transformations['Box-Cox ΔTm (Shifted)'] = pd.Series(boxcox(shifted_delta_tm)[0], index=delta_tm.index)

    # Apply Yeo-Johnson transformation, which handles both positive and negative values
    if 'Yeo-Johnson' in transformations_to_apply:
        transformations['Yeo-Johnson ΔTm'] = pd.Series(yeojohnson(delta_tm)[0], index=delta_tm.index)

    # Dictionary to store Shapiro-Wilk test results
    results = {}
    
    # Perform Shapiro-Wilk test and plot the distribution for each transformation
    for name, transformed_data in transformations.items():
        transformed_data = transformed_data.dropna()  # Drop NaN values
        
        if len(transformed_data) == 0:
            st.warning(f"No valid data available for {name} after transformation.")
            continue
        
        # Perform Shapiro-Wilk test
        stat, p_value = shapiro(transformed_data)
        results[name] = (stat, p_value)
        
        st.subheader(f"Shapiro-Wilk Test for {name}")
        st.write(f"Shapiro-Wilk Test Statistic: {stat}")
        st.write(f"P-value: {p_value}")
        
        if p_value > 0.05:
            st.write(f"The data is normally distributed after {name} (fail to reject H0).")
        else:
            st.write(f"The data is not normally distributed after {name} (reject H0).")
        
        # Plot the distribution of the transformed data
        st.subheader(f"Distribution of {name}")
        plt.figure(figsize=(6, 4))  
        sns.histplot(transformed_data, kde=True, bins=30)
        plt.title(f"Distribution of {name}")
        plt.xlabel(name)
        plt.ylabel("Frequency")
        plt.grid(True)
        st.pyplot(plt)
        plt.close()
    
    return results

def plot_melting_point_distribution(summary_table):
    st.subheader("Distribution of Melting Points")

    data = summary_table['melting point']
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

def compare_melting_points_violin(summary_table):
    st.subheader("Comparison of Melting Points Between Treatments (Interactive Violin Plot with Jittered Strip Plot)")

    # Ensure 'melting point' and 'treatment' columns exist
    if 'melting point' not in summary_table.columns or 'treatment' not in summary_table.columns:
        st.error("The summary table must contain 'melting point' and 'treatment' columns.")
        return

    # Prepare data
    data = summary_table[['melting point', 'treatment']].dropna()
    # Limit melting points to values between 0 and 100
    data = data[(data['melting point'] >= 0) & (data['melting point'] <= 100)]

    # Assign colors to treatments
    treatments = data['treatment'].unique()
    colors = px.colors.qualitative.Plotly  # Choose a color palette
    color_map = dict(zip(treatments, colors))

    # Map treatments to numerical x-values
    treatment_to_num = {treatment: idx for idx, treatment in enumerate(treatments)}

    # Create violin plots
    fig = go.Figure()

    for treatment in treatments:
        treatment_data = data[data['treatment'] == treatment]['melting point']
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

def go_annotation():
    st.title("GO Annotation Tool")

    conn = duckdb.connect('multi_proteome_go.duckdb')

    data_csv = st.file_uploader("Upload CSV file with IDs", type=['csv'])

    if data_csv is not None:
        data = pd.read_csv(data_csv)
        st.write("Preview of uploaded data:")
        st.dataframe(data.head())

        id_column = st.selectbox("Select the column containing IDs", data.columns)
        id_type = st.selectbox("Select ID Type", ["Protein", "Gene"])

        if id_column:
            species = conn.execute("SELECT name FROM species").fetchall()
            species_names = [s[0] for s in species]
            selected_species = st.selectbox("Select species for GO annotation", species_names)

            if st.button("Start Annotation"):
                ids = data[id_column].astype(str).tolist()

                annotations = get_go_annotations(conn, ids, selected_species, id_type.lower())

                # Set the name field based on the ID type
                name_field = 'Protein Name' if id_type == 'Protein' else 'Gene Name'

                data[name_field] = data[id_column].apply(
                    lambda pid: annotations.get(pid, {}).get('Name', 'NA')
                )
                data['GO ID'] = data[id_column].apply(
                    lambda pid: '|'.join(annotations.get(pid, {'GO ID': []}).get('GO ID', ['NA']))
                )
                data['Function'] = data[id_column].apply(
                    lambda pid: '|'.join(annotations.get(pid, {'Function': []}).get('Function', ['NA']))
                )
                data['Link'] = data[id_column].apply(
                    lambda pid: annotations.get(pid, {}).get('Link', 'NA')
                )

                st.subheader("Updated CSV Data with GO Annotations")
                st.dataframe(data)

                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download Annotated CSV",
                    data=csv,
                    file_name=f"annotated_{id_type.lower()}s_{selected_species.replace(' ', '_')}.csv",
                    mime='text/csv',
                )
        else:
            st.warning("Please select a column containing IDs.")

    conn.close()

def get_go_annotations(conn, ids, species_name, id_type='protein'):
    if id_type == 'protein':
        query = """
        SELECT p.id AS id, p.name AS name, g.id AS go_id, g.function, p.link
        FROM proteins p
        JOIN protein_go_terms pgt ON p.id = pgt.protein_id
        JOIN go_terms g ON pgt.go_term_id = g.id
        JOIN species s ON p.species_id = s.id
        WHERE p.id IN (SELECT CAST(unnest(?) AS VARCHAR)) AND s.name = ?
        """
    elif id_type == 'gene':
        query = """
        SELECT gn.id AS id, gn.name AS name, gt.id AS go_id, gt.function, gn.link
        FROM genes gn
        JOIN gene_go_terms ggt ON gn.id = ggt.gene_id
        JOIN go_terms gt ON ggt.go_term_id = gt.id
        JOIN species s ON gn.species_id = s.id
        WHERE gn.id IN (SELECT CAST(unnest(?) AS VARCHAR)) AND s.name = ?
        """
    else:
        raise ValueError("Invalid id_type. Must be 'protein' or 'gene'.")

    result = conn.execute(query, [ids, species_name]).fetchall()

    annotations = {}
    for row in result:
        id_, name, go_id, function, link = row
        if id_ not in annotations:
            annotations[id_] = {
                'Name': name,
                'GO ID': [],
                'Function': [],
                'Link': link
            }
        annotations[id_]['GO ID'].append(go_id)
        annotations[id_]['Function'].append(function)

    return annotations


def visualize_go_ids_with_dtm(summary_table, threshold, treatment_1, treatment_2):
    # Proceed with computations after the selections
    pivot_table = summary_table.pivot(index='protein', columns='treatment', values='melting point')

    if treatment_1 in pivot_table.columns and treatment_2 in pivot_table.columns:
        pivot_table['ΔTm'] = pivot_table[treatment_1] - pivot_table[treatment_2]
    else:
        st.error(f"The selected treatments '{treatment_1}' and '{treatment_2}' are missing in the summary table.")
        return

    filtered_table = pivot_table[pivot_table['ΔTm'].abs() > threshold].reset_index()

    if filtered_table.empty:
        st.warning(f"No proteins found with ΔTm greater than {threshold}°C.")
        return

    # Merge GO ID and Function
    go_data = summary_table[['protein', 'GO ID', 'Function']].drop_duplicates()
    filtered_table = filtered_table.merge(go_data, on='protein', how='inner')

    filtered_table = filtered_table[filtered_table['GO ID'].notna() & (filtered_table['GO ID'] != 'NA')]

    if filtered_table.empty:
        st.warning("No proteins with valid GO IDs after filtering.")
        return

    # Explode GO IDs and Functions
    filtered_table['GO ID'] = filtered_table['GO ID'].str.split(';')
    filtered_table['Function'] = filtered_table['Function'].str.split(';')
    exploded_table = filtered_table.explode(['GO ID', 'Function'])

    # Group by Function to get counts
    function_counts = exploded_table.groupby('Function').size().reset_index(name='counts')
    # Collect corresponding GO IDs
    function_go_ids = exploded_table.groupby('Function')['GO ID'].apply(lambda x: '; '.join(set(x))).reset_index()

    # Merge counts and GO IDs
    function_data = function_counts.merge(function_go_ids, on='Function')

    # Sort by counts
    function_data = function_data.sort_values(by='counts', ascending=False).head(20)

    # Wrap function names for x-axis labels
    max_label_length = 30  # Adjust as needed
    function_data['Wrapped Function'] = function_data['Function'].apply(
        lambda x: '<br>'.join(textwrap.wrap(x, width=max_label_length))
    )

    # Create interactive bar plot with Plotly
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
        xaxis_tickangle=-90,  # Rotate labels vertically
        xaxis=dict(
            automargin=True,
            tickmode='linear',
            tickfont=dict(size=10),
        ),
        margin=dict(b=150),  # Increase bottom margin to accommodate labels
    )

    st.plotly_chart(fig)

def analysis():
    st.title("TPP Analysis App")

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

    uploaded_tsv = st.file_uploader("Upload TSV fragpipe output file", type=['tsv'])
    uploaded_csv = st.file_uploader("Upload CSV metadata file", type=['csv'])

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button('Load uploaded data'):
            if uploaded_tsv is not None and uploaded_csv is not None:
                st.session_state.tsv_data = read_tsv_file(uploaded_tsv)
                st.session_state.csv_data = read_csv_file(uploaded_csv)
                st.success("Uploaded data loaded successfully!")
            else:
                st.warning("Please upload both TSV and CSV files before loading.")

    with col2:
        sample_help = "By pressing this button, sample experimental data will be loaded for demonstration purposes"
        if st.button('Load example data', help=sample_help):
            data_path = os.path.dirname(os.path.abspath(__file__))
            tsv_path = os.path.join(data_path, "sample_data.tsv")
            csv_path = os.path.join(data_path, "sample_metadata.csv")
            st.session_state.tsv_data = read_tsv_file(tsv_path)
            st.session_state.csv_data = read_csv_file(csv_path)
            st.success("Sample data loaded successfully!")

    with col3:
        if st.button('Clear loaded data'):
            st.session_state.tsv_data = None
            st.session_state.csv_data = None
            st.session_state.edit_mode_tsv = False
            st.session_state.edit_mode_csv = False
            st.success("All loaded data has been cleared!")

    if st.session_state.tsv_data is not None and st.session_state.csv_data is not None:
        st.info("TSV and CSV data are loaded and ready for analysis.")
    else:
        st.warning("Please load both TSV and CSV data to proceed with analysis.")

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

        if 'r2_threshold' not in st.session_state:
            st.session_state.r2_threshold = 0.8  # Default R² threshold

        metadata = extract_samples(st.session_state.csv_data)

        if metadata is None:
            st.error("Failed to extract samples from metadata. Please check your CSV file.")
            return
        else:
            max_allowed_zeros = st.number_input("Maximum number of missing values allowed", min_value=0, value=20, step=1)
            droppable_rows = count_invalid_rows(st.session_state.tsv_data, metadata["Samples"], max_allowed_zeros)
            st.write(f"Number of rows with {max_allowed_zeros} or more missing values: {droppable_rows} (Will be dropped)")
            
            normalize_data = st.checkbox("Normalize", value=st.session_state.get('normalize_data', True))
            
            if normalize_data:
                unique_temperatures = sorted(set(metadata['Temperature']))
                selected_temp = st.selectbox(
                    "Select the temperature to which data should be normalized:",
                    options=unique_temperatures,
                    index=unique_temperatures.index(st.session_state.get('selected_temp', unique_temperatures[0])) if st.session_state.get('selected_temp') in unique_temperatures else 0
                )
                st.session_state.normalize_data = normalize_data
                st.session_state.selected_temp = selected_temp

            else:
                selected_temp = None

            if selected_temp is None and normalize_data:
                st.warning("Please select a temperature for normalization.")

            include_go_annotation = st.checkbox("Include GO annotation", value=False)

            if include_go_annotation:
                conn = duckdb.connect('multi_proteome_go.duckdb')
                species = conn.execute("SELECT name FROM species").fetchall()
                species_names = [s[0] for s in species]
                selected_species = st.selectbox("Select species for GO annotation", species_names)
                conn.close()

            # Get the list of treatments from metadata
            treatments = list(set(metadata['Treatment']))

            with st.expander("Data Visualization Options"):
                show_distribution = st.checkbox("Show Distribution of Melting Points", value=False)
                show_violin_plot = st.checkbox("Compare Melting Points Between Treatments (Violin Plot)", value=False)
                if include_go_annotation:
                    visualize_go_ids = st.checkbox("Visualize GO IDs with ΔTm > X°C", value=False)
                    if visualize_go_ids:
                        threshold = st.number_input("ΔTm threshold for GO ID visualization", value=4.0, step=0.1)
                        if 'treatment_1' not in st.session_state:
                            st.session_state['treatment_1'] = treatments[0]
                        if 'treatment_2' not in st.session_state:
                            st.session_state['treatment_2'] = treatments[1]

                        treatment_1 = st.selectbox(
                            "Select the first treatment:",
                            treatments,
                            key='treatment_1'
                        )
                        treatment_2 = st.selectbox(
                            "Select the second treatment:",
                            treatments,
                            key='treatment_2'
                        )

            st.session_state.r2_threshold = st.slider(
                "Set the R² threshold for filtering proteins:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.r2_threshold,
                step=0.01,
            )

            with st.expander("Shapiro-Wilk Normality Test"):
                perform_shapiro = st.checkbox("Perform Shapiro-Wilk Test", value=False)
                
                if perform_shapiro:
                    st.subheader("Normality Test Selection")
                    log_transform = st.checkbox("Log Transformation", value=True)
                    sqrt_transform = st.checkbox("Square Root Transformation", value=True)
                    boxcox_transform = st.checkbox("Box-Cox Transformation", value=True)
                    yeojohnson_transform = st.checkbox("Yeo-Johnson Transformation", value=True)

                    transformations_to_apply = []
                    if log_transform:
                        transformations_to_apply.append("Log")
                    if sqrt_transform:
                        transformations_to_apply.append("Square Root")
                    if boxcox_transform:
                        transformations_to_apply.append("Box-Cox")
                    if yeojohnson_transform:
                        transformations_to_apply.append("Yeo-Johnson")

                if st.button("Apply Shapiro-Wilk Test Settings"):
                    st.session_state.perform_shapiro = perform_shapiro
                    if perform_shapiro:
                        st.session_state.transformations_to_apply = transformations_to_apply
                    st.success("Shapiro-Wilk Test settings applied successfully!")

            if st.button("Start Analysis"):
                tsv_data = st.session_state.tsv_data
                csv_data = st.session_state.csv_data

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
                    figures, summary_table = fit_and_plot(average_dict, st.session_state.selected_temp, st.session_state.normalize_data, st.session_state.r2_threshold)
                end_time = time.time()
                figure_generation_time = end_time - start_time

                st.write(f"Time taken to generate figures: {figure_generation_time:.2f} seconds |  {len(figures)} figures generated")

                if include_go_annotation:
                    conn = duckdb.connect('multi_proteome_go.duckdb')
                    with st.spinner("Adding GO annotations..."):
                        protein_ids = summary_table['protein'].tolist()
                        annotations = get_go_annotations(conn, protein_ids, selected_species)

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

                    desired_order = ['protein', 'Protein Name', 'treatment', 'melting point', 'R²', 'residuals', 'GO ID', 'Function', 'Link']
                    summary_table = summary_table[desired_order]
                
                st.subheader("Data Visualization")

                if show_distribution:
                    plot_melting_point_distribution(summary_table)

                if show_violin_plot:
                    compare_melting_points_violin(summary_table)

                if st.session_state.perform_shapiro:
                    perform_transformations_and_shapiro_test(summary_table, st.session_state['transformations_to_apply'])

                if visualize_go_ids:
                    visualize_go_ids_with_dtm(summary_table, threshold, treatment_1, treatment_2)

                start_time = time.time()

                with st.spinner('Preparing SVG files for download...'):
                    zip_file = save_as_svg(figures, summary_table)
                end_time = time.time()
                figure_save_time = end_time - start_time

                st.write(f"Time taken to save figures: {figure_save_time:.2f} seconds")

                st.download_button(
                    label="Download zipped SVGs",
                    data=zip_file,
                    file_name="protein_curves.zip",
                    mime="application/zip"
                )

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