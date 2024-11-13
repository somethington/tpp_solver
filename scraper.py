import duckdb
import requests
import time
import os
import logging
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection parameters
db_file = 'multi_proteome_go.duckdb'

def check_commit(conn, table_name):
    result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logging.info(f"After commit: {table_name} has {result} rows")
    return result

def get_go_terms_from_db(conn, go_ids):
    if not go_ids:
        return {}
    placeholders = ','.join(['?'] * len(go_ids))
    query = f"SELECT id, function FROM go_terms WHERE id IN ({placeholders})"
    result = conn.execute(query, go_ids).fetchall()
    return {row[0]: row[1] for row in result}

def get_new_go_terms(go_ids):
    url = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/" + ",".join([s.replace(":", "%3A") for s in go_ids])
    response = requests.get(url, headers={"Accept": "application/json"})
    response.raise_for_status()
    data = response.json()
    return [(item['id'], item['name']) for item in data['results']]

def get_go_ids(conn, identifier, id_type='protein'):
    url = f"https://rest.uniprot.org/uniprotkb/{identifier}?fields=go_id"
    headers = {"Accept": "application/json"}
    retries = 3

    for attempt in range(retries):
        try:
            logging.info(f"Attempting to fetch GO IDs for {id_type} {identifier} (attempt {attempt+1})")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            logging.info(f"Raw response for {id_type} {identifier}: {data}")

            # Extract GO IDs
            go_ids = []
            uni_entry = data.get('uniProtKBCrossReferences', [])
            for ref in uni_entry:
                if ref.get('database') == 'GO':
                    go_id = ref.get('id')
                    if go_id:
                        go_ids.append(go_id)

            logging.info(f"Parsed GO IDs for {id_type} {identifier}: {go_ids}")

            if not go_ids:
                logging.warning(f"No GO IDs found for {id_type} {identifier}")
                return []

            existing_terms = get_go_terms_from_db(conn, go_ids)
            new_ids = [id for id in go_ids if id not in existing_terms]

            if new_ids:
                logging.info(f"Fetching {len(new_ids)} new GO terms for {id_type} {identifier}")
                new_terms = get_new_go_terms(new_ids)
                conn.executemany('INSERT INTO go_terms (id, function) VALUES (?, ?) ON CONFLICT (id) DO NOTHING', new_terms)
                existing_terms.update(dict(new_terms))

            return [(id, existing_terms[id]) for id in go_ids]
        except requests.RequestException as e:
            logging.error(f"Error fetching GO IDs for {id_type} {identifier} (attempt {attempt+1}): {str(e)}")
            time.sleep(5)
            if attempt == retries - 1:
                logging.error(f"Failed to fetch GO IDs for {identifier} after {retries} attempts.")
                return []
        except Exception as e:
            logging.error(f"Unexpected error for {id_type} {identifier}: {str(e)}")
            return []

def extract_protein_ids_and_names(fasta_file):
    proteins = []
    fasta_file_path = os.path.join("fasta", fasta_file)
    if not os.path.exists(fasta_file_path):
        logging.error(f"File not found: {fasta_file_path}")
        return proteins

    for record in SeqIO.parse(fasta_file_path, "fasta"):
        description = record.description
        # Extract protein_id
        protein_id = description.split('|')[1]
        # Extract the rest of the description after the third '|'
        full_description = description.split('|')[2]
        # Split at ' OS='
        name_and_rest = full_description.split(' OS=')
        name_part = name_and_rest[0]
        # Split name_part at spaces
        tokens = name_part.split(' ')
        if len(tokens) > 1:
            # Protein name is tokens[1:]
            protein_name = ' '.join(tokens[1:]).strip()
        else:
            protein_name = ''
        proteins.append((protein_id, protein_name))
    return proteins

def extract_gene_ids_and_names(fasta_file):
    genes = []
    fasta_file_path = os.path.join("fasta", fasta_file)
    if not os.path.exists(fasta_file_path):
        logging.error(f"File not found: {fasta_file_path}")
        return genes

    for record in SeqIO.parse(fasta_file_path, "fasta"):
        description = record.description
        # Extract gene_id
        gene_id = description.split('|')[1]
        # Extract the rest of the description after the third '|'
        full_description = description.split('|')[2]
        # Split at ' OS='
        name_and_rest = full_description.split(' OS=')
        name_part = name_and_rest[0]
        # Split name_part at spaces
        tokens = name_part.split(' ')
        if len(tokens) > 1:
            # Gene name is tokens[1:]
            gene_name = ' '.join(tokens[1:]).strip()
        else:
            gene_name = ''
        genes.append((gene_id, gene_name))
    return genes

def extract_species_from_fasta(fasta_file):
    fasta_file_path = os.path.join("fasta", fasta_file)
    with open(fasta_file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                first_line = line.strip()
                species_start = first_line.find("OS=") + 3
                species_end = first_line.find(" OX=", species_start)
                return first_line[species_start:species_end]
    return "Unknown Species"

def create_tables(conn):
    conn.execute('''
    CREATE TABLE IF NOT EXISTS species (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE,
        proteome_file TEXT UNIQUE
    )
    ''')

    # Proteins table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS proteins (
        id TEXT PRIMARY KEY,
        species_id INTEGER,
        name TEXT,
        link TEXT,
        FOREIGN KEY (species_id) REFERENCES species (id)
    )
    ''')

    # Genes table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS genes (
        id TEXT PRIMARY KEY,
        species_id INTEGER,
        name TEXT,
        link TEXT,
        FOREIGN KEY (species_id) REFERENCES species (id)
    )
    ''')

    # GO terms table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS go_terms (
        id TEXT PRIMARY KEY,
        function TEXT
    )
    ''')

    # Protein-GO terms relationship table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS protein_go_terms (
        protein_id TEXT,
        go_term_id TEXT,
        FOREIGN KEY (protein_id) REFERENCES proteins (id),
        FOREIGN KEY (go_term_id) REFERENCES go_terms (id),
        PRIMARY KEY (protein_id, go_term_id)
    )
    ''')

    # Gene-GO terms relationship table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS gene_go_terms (
        gene_id TEXT,
        go_term_id TEXT,
        FOREIGN KEY (gene_id) REFERENCES genes (id),
        FOREIGN KEY (go_term_id) REFERENCES go_terms (id),
        PRIMARY KEY (gene_id, go_term_id)
    )
    ''')

def insert_species(conn, species_name, proteome_file):
    # Check if the species already exists
    result = conn.execute('SELECT id FROM species WHERE name = ?', (species_name,)).fetchone()
    if result:
        species_id = result[0]
    else:
        # Manually determine the next available id
        max_id = conn.execute('SELECT COALESCE(MAX(id), 0) + 1 FROM species').fetchone()[0]
        conn.execute('INSERT INTO species (id, name, proteome_file) VALUES (?, ?, ?)', (max_id, species_name, proteome_file))
        species_id = max_id
    return species_id

def insert_protein_data(conn, protein_id, species_id, go_terms, protein_name):
    # Insert or update the protein data
    link = f"https://www.uniprot.org/uniprotkb/{protein_id}"
    logging.info(f"Inserting/updating protein: {protein_id} for species: {species_id}")

    # Attempt to insert the new protein data. If the protein already exists, update the name.
    conn.execute('''
        INSERT INTO proteins (id, species_id, name, link) 
        VALUES (?, ?, ?, ?)
        ON CONFLICT (id) DO UPDATE SET name = excluded.name
    ''', (protein_id, species_id, protein_name, link))

    # Insert GO term relationships
    if go_terms:
        logging.info(f"Inserting {len(go_terms)} GO term relationships for protein: {protein_id}")
        conn.executemany('''
            INSERT INTO protein_go_terms (protein_id, go_term_id) 
            VALUES (?, ?) 
            ON CONFLICT (protein_id, go_term_id) DO NOTHING
        ''', [(protein_id, go_id) for go_id, _ in go_terms])

    protein_count = check_commit(conn, "proteins")
    go_term_count = check_commit(conn, "protein_go_terms")

    # Add total number of proteins with names
    total_names = conn.execute('SELECT COUNT(*) FROM proteins WHERE name IS NOT NULL AND name != ?', ('',)).fetchone()[0]

    logging.info(f"Committed data for protein: {protein_id}. Total proteins: {protein_count}, Total proteins with names: {total_names}, Total GO terms: {go_term_count}")

def insert_gene_data(conn, gene_id, species_id, go_terms, gene_name):
    # Insert or update the gene data
    link = f"https://www.uniprot.org/uniprotkb/{gene_id}"
    logging.info(f"Inserting/updating gene: {gene_id} for species: {species_id}")

    # Insert gene
    conn.execute('''
        INSERT INTO genes (id, species_id, name, link)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (id) DO UPDATE SET name = excluded.name
    ''', (gene_id, species_id, gene_name, link))

    # Insert GO term relationships
    if go_terms:
        logging.info(f"Inserting {len(go_terms)} GO term relationships for gene: {gene_id}")
        conn.executemany('''
            INSERT INTO gene_go_terms (gene_id, go_term_id)
            VALUES (?, ?)
            ON CONFLICT (gene_id, go_term_id) DO NOTHING
        ''', [(gene_id, go_id) for go_id, _ in go_terms])

    gene_count = check_commit(conn, "genes")
    go_term_count = check_commit(conn, "gene_go_terms")

    total_names = conn.execute('SELECT COUNT(*) FROM genes WHERE name IS NOT NULL AND name != ?', ('',)).fetchone()[0]

    logging.info(f"Committed data for gene: {gene_id}. Total genes: {gene_count}, Total genes with names: {total_names}, Total GO terms: {go_term_count}")

def get_proteins_to_process(conn, species_id, protein_ids):
    result = conn.execute('SELECT id, name FROM proteins WHERE species_id = ?', (species_id,)).fetchall()
    processed_ids_with_names = set()
    ids_missing_names = set()
    for row in result:
        protein_id, name = row
        if name and name.strip() != '':
            processed_ids_with_names.add(protein_id)
        else:
            ids_missing_names.add(protein_id)
    # Proteins not yet processed or missing names
    proteins_to_process = (set(protein_ids) - processed_ids_with_names) | ids_missing_names
    return list(proteins_to_process)

def get_genes_to_process(conn, species_id, gene_ids):
    result = conn.execute('SELECT id, name FROM genes WHERE species_id = ?', (species_id,)).fetchall()
    processed_ids_with_names = set()
    ids_missing_names = set()
    for row in result:
        gene_id, name = row
        if name and name.strip() != '':
            processed_ids_with_names.add(gene_id)
        else:
            ids_missing_names.add(gene_id)
    # Genes not yet processed or missing names
    genes_to_process = (set(gene_ids) - processed_ids_with_names) | ids_missing_names
    return list(genes_to_process)

def process_protein(protein_info, species_id, db_file):
    protein_id, protein_name = protein_info
    try:
        conn = duckdb.connect(db_file)
        go_terms = get_go_ids(conn, protein_id, id_type='protein')
        insert_protein_data(conn, protein_id, species_id, go_terms, protein_name)
        logging.info(f"Processed protein: {protein_id}, name: {protein_name}")
        return protein_id, len(go_terms)
    except Exception as e:
        logging.error(f"Error processing protein {protein_id}: {str(e)}")
        return protein_id, 0
    finally:
        conn.close()

def process_gene(gene_info, species_id, db_file):
    gene_id, gene_name = gene_info
    try:
        conn = duckdb.connect(db_file)
        go_terms = get_go_ids(conn, gene_id, id_type='gene')
        insert_gene_data(conn, gene_id, species_id, go_terms, gene_name)
        logging.info(f"Processed gene: {gene_id}, name: {gene_name}")
        return gene_id, len(go_terms)
    except Exception as e:
        logging.error(f"Error processing gene {gene_id}: {str(e)}")
        return gene_id, 0
    finally:
        conn.close()

def check_database(db_file):
    conn = duckdb.connect(db_file)
    tables = conn.execute("SHOW TABLES").fetchall()
    logging.info(f"Tables in the database: {tables}")

    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
        logging.info(f"Number of rows in {table[0]}: {count}")
    conn.close()

def main():
    fasta_directory = "fasta"
    logging.info(f"Using DuckDB database at: {db_file}")

    try:
        conn = duckdb.connect(db_file)
        create_tables(conn)

        for fasta_file in os.listdir(fasta_directory):
            if not fasta_file.endswith('.fasta'):
                continue

            species_name = extract_species_from_fasta(fasta_file)
            species_id = insert_species(conn, species_name, fasta_file)

            logging.info(f"Processing {fasta_file} for species: {species_name} (ID: {species_id})")

            # Process Proteins
            proteins = extract_protein_ids_and_names(fasta_file)
            protein_dict = {protein_id: protein_name for protein_id, protein_name in proteins}
            protein_ids = list(protein_dict.keys())

            proteins_to_process_ids = get_proteins_to_process(conn, species_id, protein_ids)
            proteins_to_process = [(protein_id, protein_dict[protein_id]) for protein_id in proteins_to_process_ids]

            logging.info(f"Found {len(protein_ids)} proteins in {fasta_file}")
            logging.info(f"Processing {len(proteins_to_process)} proteins (unprocessed or missing names)")

            start_time = time.time()

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_protein, protein_info, species_id, db_file) for protein_info in proteins_to_process]

                for future in as_completed(futures):
                    protein_id, go_term_count = future.result()
                    logging.info(f"Completed processing protein: {protein_id} with {go_term_count} GO terms")

            end_time = time.time()
            processing_time = end_time - start_time

            if len(proteins_to_process) > 0:
                logging.info(f"Processed {len(proteins_to_process)} proteins in {processing_time:.2f} seconds")
                logging.info(f"Average time per protein: {processing_time / len(proteins_to_process):.2f} seconds")

            # Verify protein insertion
            protein_count = conn.execute('SELECT COUNT(*) FROM proteins WHERE species_id = ?', (species_id,)).fetchone()[0]
            proteins_with_names = conn.execute('SELECT COUNT(*) FROM proteins WHERE species_id = ? AND name IS NOT NULL AND name != ?', (species_id, '')).fetchone()[0]
            logging.info(f"Total proteins in database for species {species_name}: {protein_count}")
            logging.info(f"Total proteins with names for species {species_name}: {proteins_with_names}")

            final_protein_count = check_commit(conn, "proteins")
            final_go_term_count = check_commit(conn, "protein_go_terms")
            logging.info(f"Completed processing proteins in {fasta_file}. Final counts - Proteins: {final_protein_count}, GO terms: {final_go_term_count}")

            # Process Genes
            genes = extract_gene_ids_and_names(fasta_file)
            gene_dict = {gene_id: gene_name for gene_id, gene_name in genes}
            gene_ids = list(gene_dict.keys())

            genes_to_process_ids = get_genes_to_process(conn, species_id, gene_ids)
            genes_to_process = [(gene_id, gene_dict[gene_id]) for gene_id in genes_to_process_ids]

            logging.info(f"Found {len(gene_ids)} genes in {fasta_file}")
            logging.info(f"Processing {len(genes_to_process)} genes (unprocessed or missing names)")

            start_time = time.time()

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_gene, gene_info, species_id, db_file) for gene_info in genes_to_process]

                for future in as_completed(futures):
                    gene_id, go_term_count = future.result()
                    logging.info(f"Completed processing gene: {gene_id} with {go_term_count} GO terms")

            end_time = time.time()
            processing_time = end_time - start_time

            if len(genes_to_process) > 0:
                logging.info(f"Processed {len(genes_to_process)} genes in {processing_time:.2f} seconds")
                logging.info(f"Average time per gene: {processing_time / len(genes_to_process):.2f} seconds")

            # Verify gene insertion
            gene_count = conn.execute('SELECT COUNT(*) FROM genes WHERE species_id = ?', (species_id,)).fetchone()[0]
            genes_with_names = conn.execute('SELECT COUNT(*) FROM genes WHERE species_id = ? AND name IS NOT NULL AND name != ?', (species_id, '')).fetchone()[0]
            logging.info(f"Total genes in database for species {species_name}: {gene_count}")
            logging.info(f"Total genes with names for species {species_name}: {genes_with_names}")

            final_gene_count = check_commit(conn, "genes")
            final_go_term_count = check_commit(conn, "gene_go_terms")
            logging.info(f"Completed processing genes in {fasta_file}. Final counts - Genes: {final_gene_count}, GO terms: {final_go_term_count}")

        # At the end of processing, check overall database status
        total_proteins = conn.execute('SELECT COUNT(*) FROM proteins').fetchone()[0]
        total_proteins_with_names = conn.execute('SELECT COUNT(*) FROM proteins WHERE name IS NOT NULL AND name != ?', ('',)).fetchone()[0]
        total_protein_go_terms = conn.execute('SELECT COUNT(*) FROM protein_go_terms').fetchone()[0]
        total_genes = conn.execute('SELECT COUNT(*) FROM genes').fetchone()[0]
        total_genes_with_names = conn.execute('SELECT COUNT(*) FROM genes WHERE name IS NOT NULL AND name != ?', ('',)).fetchone()[0]
        total_gene_go_terms = conn.execute('SELECT COUNT(*) FROM gene_go_terms').fetchone()[0]
        logging.info(f"Total proteins in database: {total_proteins}")
        logging.info(f"Total proteins with names in database: {total_proteins_with_names}")
        logging.info(f"Total protein-GO term relationships: {total_protein_go_terms}")
        logging.info(f"Total genes in database: {total_genes}")
        logging.info(f"Total genes with names in database: {total_genes_with_names}")
        logging.info(f"Total gene-GO term relationships: {total_gene_go_terms}")

    except Exception as e:
        logging.error(f"Database error: {e}")
    finally:
        conn.close()

    # Final database check
    check_database(db_file)

    logging.info('All processing complete')

if __name__ == "__main__":
    main()
