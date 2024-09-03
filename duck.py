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

def get_ids_names(conn, protein):
    url = f"https://rest.uniprot.org/uniprotkb/{protein}?fields=go_id"
    headers = {"Accept": "text/plain; format=tsv"}
    retries = 3

    for attempt in range(retries):
        try:
            logging.info(f"Attempting to fetch data for protein {protein} (attempt {attempt+1})")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            logging.info(f"Raw response for protein {protein}: {response.text}")
            go_ids = list(set([s.replace(";", "") for s in response.text.split()[3:]]))  # Use set to remove duplicates
            logging.info(f"Parsed unique GO IDs for protein {protein}: {go_ids}")

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
            logging.error(f"Error fetching data for protein {protein} (attempt {attempt+1}): {str(e)}")
            time.sleep(5)
            if attempt == retries - 1:
                logging.error(f"Failed to fetch data for {protein} after {retries} attempts.")
                return []

def extract_protein_ids(fasta_file):
    protein_ids = []
    fasta_file_path = os.path.join("fasta", fasta_file)
    if not os.path.exists(fasta_file_path):
        logging.error(f"File not found: {fasta_file_path}")
        return protein_ids

    for record in SeqIO.parse(fasta_file_path, "fasta"):
        description = record.description
        protein_id = description.split('|')[1]
        protein_ids.append(protein_id)
    return protein_ids

def extract_species_from_fasta(fasta_file):
    with open(fasta_file, 'r') as file:
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
    conn.execute('''
    CREATE TABLE IF NOT EXISTS proteins (
        id TEXT PRIMARY KEY,
        species_id INTEGER,
        link TEXT,
        FOREIGN KEY (species_id) REFERENCES species (id)
    )
    ''')
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

def insert_protein_data(conn, protein_id, species_id, go_terms):
    # Insert or ignore if the protein already exists
    link = f"https://www.uniprot.org/uniprotkb/{protein_id}"
    logging.info(f"Inserting/updating protein: {protein_id} for species: {species_id}")
    
    # Attempt to insert the new protein data. If the protein already exists, the insert will be ignored.
    conn.execute('''
        INSERT INTO proteins (id, species_id, link) 
        VALUES (?, ?, ?)
        ON CONFLICT (id) DO NOTHING
    ''', (protein_id, species_id, link))

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
    logging.info(f"Committed data for protein: {protein_id}. Total proteins: {protein_count}, Total GO terms: {go_term_count}")

def get_processed_ids(conn, species_id):
    result = conn.execute('SELECT id FROM proteins WHERE species_id = ?', (species_id,)).fetchall()
    return set(row[0] for row in result)

def process_protein(protein, species_id, db_file):
    try:
        conn = duckdb.connect(db_file)
        go_terms = get_ids_names(conn, protein)
        insert_protein_data(conn, protein, species_id, go_terms)
        logging.info(f"Processed protein: {protein}")
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

            fasta_path = os.path.join(fasta_directory, fasta_file)
            species_name = extract_species_from_fasta(fasta_path)
            species_id = insert_species(conn, species_name, fasta_file)

            logging.info(f"Processing {fasta_file} for species: {species_name} (ID: {species_id})")

            processed_ids = get_processed_ids(conn, species_id)
            protein_ids = extract_protein_ids(fasta_file)

            logging.info(f"Found {len(protein_ids)} proteins in {fasta_file}")
            logging.info(f"{len(processed_ids)} proteins already processed")

            unprocessed_proteins = [p for p in protein_ids if p not in processed_ids]
            logging.info(f"Processing {len(unprocessed_proteins)} new proteins")

            start_time = time.time()

            # Use ThreadPoolExecutor for concurrent processing
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_protein, protein, species_id, db_file) for protein in unprocessed_proteins]
                
                for future in as_completed(futures):
                    protein, go_term_count = future.result()
                    logging.info(f"Completed processing protein: {protein} with {go_term_count} GO terms")

            end_time = time.time()
            processing_time = end_time - start_time

            if len(unprocessed_proteins) > 0:
                logging.info(f"Processed {len(unprocessed_proteins)} proteins in {processing_time:.2f} seconds")
                logging.info(f"Average time per protein: {processing_time / len(unprocessed_proteins):.2f} seconds")

            # After processing all proteins for a species, verify the insertion
            protein_count = conn.execute('SELECT COUNT(*) FROM proteins WHERE species_id = ?', (species_id,)).fetchone()[0]
            logging.info(f"Total proteins in database for species {species_name}: {protein_count}")

            final_protein_count = check_commit(conn, "proteins")
            final_go_term_count = check_commit(conn, "protein_go_terms")
            logging.info(f"Completed processing {fasta_file}. Final counts - Proteins: {final_protein_count}, GO terms: {final_go_term_count}")

        # At the end of processing, check overall database status
        total_proteins = conn.execute('SELECT COUNT(*) FROM proteins').fetchone()[0]
        total_go_terms = conn.execute('SELECT COUNT(*) FROM protein_go_terms').fetchone()[0]
        logging.info(f"Total proteins in database: {total_proteins}")
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