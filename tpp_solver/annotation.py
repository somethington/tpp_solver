"""Attach GO annotations (names, functions, links) to result tables."""
import streamlit as st

from .database import get_db_connection, get_go_annotations

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
    conn = get_db_connection()
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

def add_go_annotations(summary_table, selected_species):
    """
    Add GO annotations to the summary table.
    
    Args:
        summary_table (pd.DataFrame): DataFrame containing the protein analysis results
        selected_species (str): Name of the selected species for GO annotation
        
    Returns:
        pd.DataFrame: Summary table with added GO annotations
    """
    conn = get_db_connection()

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

    # Reorder columns to keep consistent format
    desired_order = ['protein', 'Protein Name', 'treatment', 'melting_point', 
                    'R²', 'residuals', 'GO ID', 'Function', 'Link']
    summary_table = summary_table[desired_order]
    
    return summary_table
