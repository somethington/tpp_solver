"""Plotly/Matplotlib visualizations rendered into the Streamlit app."""
import textwrap

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde

from .preprocessing import calculate_bin_width

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
    color_map = dict(zip(treatments, colors, strict=False))
    treatment_to_num = {treatment: idx for idx, treatment in enumerate(treatments)}

    # Create violin plots
    fig = go.Figure()

    # Deterministic jitter so the plot is reproducible across reruns.
    jitter_rng = np.random.default_rng(0)

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
        jittered_x = treatment_num + jitter_rng.uniform(-jitter_strength, jitter_strength, size=len(treatment_data))

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
