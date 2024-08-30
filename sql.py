import sqlite3
import requests
import time
import os
from Bio import SeqIO

def get_go_terms_from_db(conn, go_ids):
    cursor = conn.cursor()
    placeholders = ','.join(['?' for _ in go_ids])
    cursor.execute(f"SELECT id, function FROM go_terms WHERE id IN ({placeholders})", go_ids)
    return {row[0]: row[1] for row in cursor.fetchall()}

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
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            go_ids = [s.replace(";", "") for s in response.text.split()[3:]]
            if not go_ids:
                return []

            # Check which GO terms we already have in the database
            existing_terms = get_go_terms_from_db(conn, go_ids)
            new_ids = [id for id in go_ids if id not in existing_terms]

            if new_ids:
                new_terms = get_new_go_terms(new_ids)
                # Insert new terms into the database
                cursor = conn.cursor()
                cursor.executemany('INSERT OR IGNORE INTO go_terms (id, function) VALUES (?, ?)', new_terms)
                conn.commit()
                existing_terms.update(dict(new_terms))

            return [(id, existing_terms[id]) for id in go_ids]
        except requests.RequestException:
            time.sleep(5)
            if attempt == retries - 1:
                print(f"Failed to fetch data for {protein} after {retries} attempts.")
                return []

def extract_protein_ids(fasta_file):
    protein_ids = []
    fasta_file_path = os.path.join("fasta", fasta_file)
    if not os.path.exists(fasta_file_path):
        print(f"File not found: {fasta_file_path}")
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
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS species (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        proteome_file TEXT UNIQUE
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS proteins (
        id TEXT PRIMARY KEY,
        species_id INTEGER,
        link TEXT,
        FOREIGN KEY (species_id) REFERENCES species (id)
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS go_terms (
        id TEXT PRIMARY KEY,
        function TEXT
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS protein_go_terms (
        protein_id TEXT,
        go_term_id TEXT,
        FOREIGN KEY (protein_id) REFERENCES proteins (id),
        FOREIGN KEY (go_term_id) REFERENCES go_terms (id),
        PRIMARY KEY (protein_id, go_term_id)
    )
    ''')
    conn.commit()

def insert_species(conn, species_name, proteome_file):
    cursor = conn.cursor()
    cursor.execute('INSERT OR IGNORE INTO species (name, proteome_file) VALUES (?, ?)', (species_name, proteome_file))
    conn.commit()
    cursor.execute('SELECT id FROM species WHERE name = ?', (species_name,))
    return cursor.fetchone()[0]

def insert_protein_data(conn, protein_id, species_id, go_terms):
    cursor = conn.cursor()
    
    # Insert protein
    link = f"https://www.uniprot.org/uniprotkb/{protein_id}"
    cursor.execute('INSERT OR REPLACE INTO proteins (id, species_id, link) VALUES (?, ?, ?)', (protein_id, species_id, link))
    
    # Insert GO term relationships
    for go_id, _ in go_terms:
        cursor.execute('INSERT OR IGNORE INTO protein_go_terms (protein_id, go_term_id) VALUES (?, ?)', (protein_id, go_id))
    
    conn.commit()

def get_processed_ids(conn, species_id):
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM proteins WHERE species_id = ?', (species_id,))
    return set(row[0] for row in cursor.fetchall())

def main():
    fasta_directory = "fasta"
    db_path = os.path.join(fasta_directory, "multi_proteome_go.db")

    with sqlite3.connect(db_path) as conn:
        create_tables(conn)

        for fasta_file in os.listdir(fasta_directory):
            if not fasta_file.endswith('.fasta'):
                continue

            fasta_path = os.path.join(fasta_directory, fasta_file)
            species_name = extract_species_from_fasta(fasta_path)
            species_id = insert_species(conn, species_name, fasta_file)

            processed_ids = get_processed_ids(conn, species_id)
            protein_ids = extract_protein_ids(fasta_file)

            print(f"Processing {fasta_file} for species: {species_name}")

            for protein in protein_ids:
                if protein in processed_ids:
                    continue  # Skip already processed proteins

                go_terms = get_ids_names(conn, protein)
                
                # Insert data into the database
                insert_protein_data(conn, protein, species_id, go_terms)
                
                # Print progress
                print(f"Processed: {protein} - GO terms: {len(go_terms)}")

            print(f'Processing complete for {fasta_file}')

    print('All processing complete')

if __name__ == "__main__":
    main()