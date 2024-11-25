import duckdb
import requests
import time
import os
import logging
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection parameters
db_file = 'multi_proteome_go.duckdb'

def check_commit(conn, table_name):
    result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logging.info(f"After commit: {table_name} has {result} rows")
    return result

def get_go_terms_from_db(conn, go_ids):
    placeholders = ','.join(['?'] * len(go_ids))
    result = conn.execute(f"SELECT id, function FROM go_terms WHERE id IN ({placeholders})", go_ids).fetchall()
    return {row[0]: row[1] for row in result}

def get_new_go_terms(go_ids):
    url = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/" + ",".join([s.replace(":", "%3A") for s in go_ids])
    response = requests.get(url, headers={"Accept": "application/json"})
    response.raise_for_status()
    data = response.json()
    return [(item['id'], item['name']) for item in data['results']]

def get_go_ids(conn, protein):
    url = f"https://rest.uniprot.org/uniprotkb/{protein}?fields=go_id"
    headers = {"Accept": "application/json"}
    retries = 3

    for attempt in range(retries):
        try:
            logging.info(f"Attempting to fetch GO IDs for protein {protein} (attempt {attempt+1})")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            logging.info(f"Raw response for protein {protein}: {data}")

            # Extract GO IDs
            go_ids = data.get('goId', [])
            logging.info(f"Parsed GO IDs for protein {protein}: {go_ids}")

            if not go_ids:
                logging.warning(f"No GO IDs found for protein {protein}")
                return []

            existing_terms = get_go_terms_from_db(conn, go_ids)
            new_ids = [id for id in go_ids if id not in existing_terms]

            if new_ids:
                logging.info(f"Fetching {len(new_ids)} new GO terms for protein {protein}")
                new_terms = get_new_go_terms(new_ids)
                conn.executemany('INSERT INTO go_terms (id, function) VALUES (?, ?) ON CONFLICT (id) DO NOTHING', new_terms)
                existing_terms.update(dict(new_terms))

            return [(id, existing_terms[id]) for id in go_ids]
        except requests.RequestException as e:
            logging.error(f"Error fetching GO IDs for protein {protein} (attempt {attempt+1}): {str(e)}")
            time.sleep(5)
            if attempt == retries - 1:
                logging.error(f"Failed to fetch GO IDs for {protein} after {retries} attempts.")
                return []
        except Exception as e:
            logging.error(f"Unexpected error for protein {protein}: {str(e)}")
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

def extract_species_from_fasta(fasta_file):
    fasta_file_path = os.path.join("fasta", fasta_file)
    with open(fasta_file_path, 'r') as file:
        first_line = file.readline().strip()
        species_start = first_line.find("OS=") + 3
        species_end = first_line.find(" OX=", species_start)
        return first_line[species_start:species_end]

def create_tables(conn):
    conn.execute('''
    CREATE TABLE IF NOT EXISTS species (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE,
        proteome_file TEXT UNIQUE
    )
    ''')

    # Check if 'proteins' table exists
    tables = conn.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]

    if 'proteins' in table_names:
        # Check if 'name' column exists
        columns = conn.execute("PRAGMA table_info('proteins')").fetchall()
        column_names = [col[1] for col in columns]
        if 'name' not in column_names:
            conn.execute('ALTER TABLE proteins ADD COLUMN name TEXT')
            logging.info("Added 'name' column to 'proteins' table")
        else:
            logging.info("'name' column already exists in 'proteins' table")
    else:
        # Create 'proteins' table with 'name' column
        conn.execute('''
        CREATE TABLE proteins (
            id TEXT PRIMARY KEY,
            species_id INTEGER,
            name TEXT,
            link TEXT,
            FOREIGN KEY (species_id) REFERENCES species (id)
        )
        ''')

    # Create other tables as before
    conn.execute('''
    CREATE TABLE IF NOT EXISTS go_terms (
        id TEXT PRIMARY KEY,
        function TEXT
    )
    ''')
    conn.execute('''
    CREATE TABLE IF NOT EXISTS protein_go_terms (
        protein_id TEXT,
        go_term_id TEXT,
        FOREIGN KEY (protein_id) REFERENCES proteins (id),
        FOREIGN KEY (go_term_id) REFERENCES go_terms (id),
        PRIMARY KEY (protein_id, go_term_id)
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

def process_protein(protein_info, species_id, db_file):
    protein, protein_name = protein_info
    try:
        conn = duckdb.connect(db_file)
        go_terms = get_go_ids(conn, protein)
        insert_protein_data(conn, protein, species_id, go_terms, protein_name)
        logging.info(f"Processed protein: {protein}, name: {protein_name}")
        return protein, len(go_terms)
    except Exception as e:
        logging.error(f"Error processing protein {protein}: {str(e)}")
        return protein, 0
    finally:
        conn.close()

def check_database(db_file):
    conn = duckdb.connect(db_file)
    tables = conn.execute("SHOW TABLES").fetchall()
    logging.info(f"Tables in the database: {tables}")

    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
        logging.info(f"Number of rows in {table[0]}: {count}")

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

            # Extract proteins with IDs and names
            proteins = extract_protein_ids_and_names(fasta_file)
            # Create a dictionary mapping IDs to names
            protein_dict = {protein_id: protein_name for protein_id, protein_name in proteins}
            protein_ids = list(protein_dict.keys())

            # Get the proteins to process
            proteins_to_process_ids = get_proteins_to_process(conn, species_id, protein_ids)

            # Create a list of (protein_id, protein_name) for proteins to process
            proteins_to_process = [(protein_id, protein_dict[protein_id]) for protein_id in proteins_to_process_ids]

            logging.info(f"Found {len(protein_ids)} proteins in {fasta_file}")
            logging.info(f"Processing {len(proteins_to_process)} proteins (unprocessed or missing names)")

            start_time = time.time()

            # Use ThreadPoolExecutor for concurrent processing
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_protein, protein_info, species_id, db_file) for protein_info in proteins_to_process]

                for future in as_completed(futures):
                    protein, go_term_count = future.result()
                    logging.info(f"Completed processing protein: {protein} with {go_term_count} GO terms")

            end_time = time.time()
            processing_time = end_time - start_time

            if len(proteins_to_process) > 0:
                logging.info(f"Processed {len(proteins_to_process)} proteins in {processing_time:.2f} seconds")
                logging.info(f"Average time per protein: {processing_time / len(proteins_to_process):.2f} seconds")

            # After processing all proteins for a species, verify the insertion
            protein_count = conn.execute('SELECT COUNT(*) FROM proteins WHERE species_id = ?', (species_id,)).fetchone()[0]
            proteins_with_names = conn.execute('SELECT COUNT(*) FROM proteins WHERE species_id = ? AND name IS NOT NULL AND name != ?', (species_id, '')).fetchone()[0]
            logging.info(f"Total proteins in database for species {species_name}: {protein_count}")
            logging.info(f"Total proteins with names for species {species_name}: {proteins_with_names}")

            final_protein_count = check_commit(conn, "proteins")
            final_go_term_count = check_commit(conn, "protein_go_terms")
            logging.info(f"Completed processing {fasta_file}. Final counts - Proteins: {final_protein_count}, GO terms: {final_go_term_count}")

        # At the end of processing, check overall database status
        total_proteins = conn.execute('SELECT COUNT(*) FROM proteins').fetchone()[0]
        total_proteins_with_names = conn.execute('SELECT COUNT(*) FROM proteins WHERE name IS NOT NULL AND name != ?', ('',)).fetchone()[0]
        total_go_terms = conn.execute('SELECT COUNT(*) FROM protein_go_terms').fetchone()[0]
        logging.info(f"Total proteins in database: {total_proteins}")
        logging.info(f"Total proteins with names in database: {total_proteins_with_names}")
        logging.info(f"Total protein-GO term relationships: {total_go_terms}")

    except Exception as e:
        logging.error(f"Database error: {e}")
    finally:
        conn.close()

    # Final database check
    check_database(db_file)

    logging.info('All processing complete')

if __name__ == "__main__":
    main()