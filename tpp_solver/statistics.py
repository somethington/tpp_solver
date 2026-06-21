"""Statistical tests and normality diagnostics for melting points."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import boxcox, gaussian_kde, mannwhitneyu, shapiro, yeojohnson

from .preprocessing import calculate_bin_width


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR-adjusted p-values, returned in the input order.

    Args:
        p_values: sequence of raw p-values.

    Returns:
        numpy array of FDR values aligned with the input order.
    """
    p_values = np.asarray(p_values, dtype=float)
    n_tests = len(p_values)
    if n_tests == 0:
        return np.array([])

    order = np.argsort(p_values)
    ranked = p_values[order]
    fdr = ranked * n_tests / (np.arange(n_tests) + 1)

    # Enforce monotonicity from the largest p-value downward.
    for i in range(n_tests - 2, -1, -1):
        fdr[i] = min(fdr[i], fdr[i + 1])

    # Restore original ordering.
    out = np.empty(n_tests)
    out[order] = fdr
    return out

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
    
    # Calculate Benjamini-Hochberg FDR-adjusted p-values
    results_df['FDR'] = benjamini_hochberg(raw_p_values)
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
        y=-np.log10(results_df['FDR']),  
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
            "-log10(FDR): %{y:.2f}<br>" + 
            "<extra></extra>"
        )
    ))
    
    # Add significance threshold line
    fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red", opacity=0.5)
    
    fig.update_layout(
        title="Volcano Plot of Melting Point Changes",
        xaxis_title="ΔTm (°C)",
        yaxis_title="-log10(FDR)",  
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig)
