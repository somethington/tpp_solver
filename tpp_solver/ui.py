"""Streamlit user interface and analysis orchestration."""
import io
import os
import time
import zipfile
import multiprocessing as mp

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from .annotation import add_go_annotations, annotate_proteins
from .database import get_species_list
from .fitting import fit_and_plot_averaged_curves, fit_and_plot_replicates
from .io_utils import read_csv_file, read_tsv_file, save_figure_to_svg
from .plotting import (
    compare_melting_points_violin,
    plot_melting_point_distribution,
    visualize_go_ids_with_dtm,
)
from .preprocessing import (
    filter_and_lowest_float,
    get_replicate_data,
    impute_filtered_data,
    process_summary_tables,
)
from .readme_content import display_readme
from .statistics import (
    perform_protein_statistical_tests,
    perform_transformations_and_shapiro_test,
)

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
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = 42

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
        # Add normalization method selector
        norm_methods = [
            "Reference Temperature",
            "Median",
            "Robust Z-score",
            "Quantile",
            "Winsorization"
        ]
        
        selected_method = st.selectbox(
            "Select normalization method:",
            options=norm_methods,
            help="""
            - Reference Temperature: Normalize to a specific temperature point
            - Median: Robust to outliers, accounts for loading differences
            - Robust Z-score: Uses median and MAD, highly resistant to outliers
            - Quantile: Forces identical distributions across samples
            - Winsorization: Limits extreme values while preserving data structure
            """
        )
        
        st.session_state.norm_method = selected_method
        
        # Show temperature selector only for reference temperature normalization
        if selected_method == "Reference Temperature":
            unique_temperatures = sorted(set(metadata['Temperature']))
            selected_temp = st.selectbox(
                "Select the reference temperature:",
                options=unique_temperatures,
                index=unique_temperatures.index(st.session_state.get('selected_temp', unique_temperatures[0])) 
                if st.session_state.get('selected_temp') in unique_temperatures else 0
            )
            st.session_state.selected_temp = selected_temp
        else:
            st.session_state.selected_temp = None
            
        # Show winsorization limits if that method is selected
        if selected_method == "Winsorization":
            limits = st.slider(
                "Set winsorization limits (percentiles):",
                min_value=0.0,
                max_value=0.5,
                value=(0.05, 0.95),
                step=0.05,
                help="Data points outside these percentiles will be capped"
            )
            st.session_state.winsor_limits = limits
    else:
        st.session_state.selected_temp = None
        st.session_state.norm_method = None
        
    st.session_state.normalize_data = normalize_data
    
    if st.session_state.selected_temp is None and normalize_data and st.session_state.norm_method == "Reference Temperature":
        st.warning("Please select a reference temperature for normalization.")
        
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
        species_names = get_species_list()
        selected_species = st.selectbox("Select species for GO annotation", species_names)

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
                
            # Records the normalization choice in st.session_state for later use.
            setup_analysis_parameters(metadata)

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

            # Random seed for reproducible imputation of missing values
            st.session_state.random_seed = st.number_input(
                "Random seed (for reproducible imputation):",
                min_value=0,
                value=int(st.session_state.get('random_seed', 42)),
                step=1,
                help="Imputation fills missing values with random draws. Fixing the "
                     "seed makes results reproducible across runs."
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
                        ceiling_rand,
                        seed=st.session_state.get('random_seed', 42)
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
        page_icon="assets/logo_32x32.png",
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
