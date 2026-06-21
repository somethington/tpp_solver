"""DuckDB access for GO/proteome annotations."""
import os

import duckdb
import streamlit as st

from . import PROJECT_ROOT

# Path to the bundled GO/proteome database, resolved relative to the project root
# so the app works regardless of the process working directory.
DB_PATH = os.path.join(PROJECT_ROOT, "multi_proteome_go.duckdb")

@st.cache_resource
def get_db_connection():
    """Return a cached, read-only DuckDB connection.

    Streamlit re-runs the whole script on every interaction; caching the
    connection avoids reopening the database on each rerun. Read-only mode also
    lets multiple app replicas share the same database file.
    """
    return duckdb.connect(DB_PATH, read_only=True)

@st.cache_data
def get_species_list():
    """
    Get list of available species from the database.

    Returns:
        list: List of species names
    """
    conn = get_db_connection()
    species = conn.execute("SELECT name FROM species").fetchall()
    return [s[0] for s in species]

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
