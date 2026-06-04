"""Sigmoidal curve fitting and figure generation (single and averaged)."""
import itertools
import multiprocessing as mp

import matplotlib
matplotlib.use("Agg")  # headless/multiprocessing-safe backend (no GUI required)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit

from .database import get_db_connection, get_go_annotations
from .models import sigmoid
from .normalization import (
    normalize_by_median,
    normalize_by_quantile,
    normalize_by_robust_zscore,
    normalize_by_winsorization,
)
from .preprocessing import _slice_replicate_data

def process_protein_replicates(args):
    """
    Process individual protein replicates for curve fitting with memory efficiency.
    
    Returns:
        tuple: (protein, fig, summary_data) if successful, (None, None, None) if not
    """
    (
        protein,
        data_dict,
        marker_opts,
        size_opts,
        alpha_opts,
        selected_temp,
        normalize_data,
        r2_threshold,
        norm_method,
        winsor_limits,
    ) = args

    # Build styling cycles locally; this runs in a worker process and must not
    # share mutable iterators or touch Streamlit's session state.
    markers = itertools.cycle(marker_opts)
    sizes = itertools.cycle(size_opts)
    alphas = itertools.cycle(alpha_opts)

    summary_data = []
    fig = None
    ax = None

    try:
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
                    
                    if normalize_data:
                        if norm_method == "Reference Temperature" and selected_temp in temperatures:
                            norm_idx = np.where(temperatures == selected_temp)[0][0]
                            norm_value = values[norm_idx]
                            if norm_value != 0:  # Avoid division by zero
                                values = values / norm_value
                        elif norm_method == "Median":
                            values = normalize_by_median(values)
                        elif norm_method == "Robust Z-score":
                            values = normalize_by_robust_zscore(values)
                        elif norm_method == "Quantile":
                            values = normalize_by_quantile(values)
                        elif norm_method == "Winsorization":
                            values = normalize_by_winsorization(values, limits=winsor_limits)
                    
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

    # Plot styling options; each worker builds its own cycles to stay process-safe.
    marker_opts = ['o', 's', '^', 'v']
    size_opts = [50, 75, 100]
    alpha_opts = [1.0, 0.8, 0.6]

    # Resolve normalization settings on the main thread. Worker processes have no
    # Streamlit session context (the default start method is 'spawn' on macOS and
    # Windows), so these must be passed explicitly rather than read from st.*.
    norm_method = st.session_state.get('norm_method', 'Reference Temperature')
    winsor_limits = st.session_state.get('winsor_limits', (0.05, 0.95))

    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_proteins = len(all_proteins)

    # Prepare arguments for parallel processing. Each worker gets ONLY its
    # protein's slice of the data, not the whole dataset.
    process_args = [
        (
            protein,
            _slice_replicate_data(replicate_data, protein),
            marker_opts,
            size_opts,
            alpha_opts,
            selected_temp,
            normalize_data,
            r2_threshold,
            norm_method,
            winsor_limits,
        )
        for protein in all_proteins
    ]

    # Process results with progress bar
    figures = {}
    all_summary_data = []

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for i, result in enumerate(pool.imap(process_protein_replicates, process_args)):
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

def fit_and_plot_averaged_curves(replicate_data, selected_temp=None, normalize_data=True, r2_threshold=0.8, selected_species=None, include_go_annotation=False):
    """
    Fit sigmoidal curves to averaged data with spaced melting point markers.
    """
    figures = {}
    summary_data = []
    all_proteins = set()
    
    for proteins_dict in replicate_data.values():
        for prot in proteins_dict.keys():
            all_proteins.add(prot)

    total_proteins = len(all_proteins)
    progress_bar = st.progress(0)
    status_text = st.empty()
    st.write("Fitting averaged curves...")

    # Resolve normalization settings once (this function runs on the main thread).
    norm_method = st.session_state.get('norm_method', 'Reference Temperature')
    winsor_limits = st.session_state.get('winsor_limits', (0.05, 0.95))

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

            if normalize_data:
                if norm_method == "Reference Temperature" and selected_temp in temperatures:
                    norm_idx = np.where(temperatures == selected_temp)[0][0]
                    norm_value = values[norm_idx]
                    if norm_value != 0:
                        values = values / norm_value
                elif norm_method == "Median":
                    values = normalize_by_median(values)
                elif norm_method == "Robust Z-score":
                    values = normalize_by_robust_zscore(values)
                elif norm_method == "Quantile":
                    values = normalize_by_quantile(values)
                elif norm_method == "Winsorization":
                    values = normalize_by_winsorization(values, limits=winsor_limits)

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
            conn = get_db_connection()
            st.write("Adding GO annotations to averaged summary table...")
            protein_ids = averaged_summary_table['protein'].unique().tolist()
            annotations = get_go_annotations(conn, protein_ids, selected_species)

            for col, default in [('GO ID', ['NA']), ('Function', ['NA']),
                               ('Protein Name', 'NA'), ('Link', 'NA')]:
                averaged_summary_table[col] = averaged_summary_table['protein'].apply(
                    lambda pid, col=col, default=default: ';'.join(annotations.get(pid, {col: default})[col])
                    if isinstance(default, list)
                    else annotations.get(pid, {col: default})[col]
                )


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
