from Bio import SeqIO
import pandas as pd
import requests
import time
import os


def get_ids_names(protein):
    url = f"https://rest.uniprot.org/uniprotkb/{protein}?fields=go_id"
    headers = {"Accept": "text/plain; format=tsv"}
    retries = 3  # Max retries for each request

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            res_list = [s.replace(";", "") for s in response.text.split()[3:]]
            if not res_list:
                return "NA", "NA"

            url = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/" + ",".join([s.replace(":", "%3A") for s in res_list])
            response = requests.get(url, headers={"Accept": "application/json"})
            response.raise_for_status()
            data = response.json()

            ids = "|".join([item['id'] for item in data['results']])
            funcs = "|".join([item['name'] for item in data['results']])
            return ids, funcs
        except requests.RequestException:
            time.sleep(5)  # Wait for 5 seconds before retrying
            if attempt == retries - 1:
                print(f"Failed to fetch data for {protein} after {retries} attempts.")
                return "NA", "NA"

def extract_protein_ids(fasta_file):
    protein_ids = []
    fasta_file_path = os.path.join("fasta", fasta_file)  # Construct the full path to the FASTA file
    if not os.path.exists(fasta_file_path):
        print(f"File not found: {fasta_file_path}")
        return protein_ids  # Return an empty list if the file is not found

    for record in SeqIO.parse(fasta_file_path, "fasta"):
        description = record.description
        protein_id = description.split('|')[1]
        protein_ids.append(protein_id)
    return protein_ids

def main():
    fasta_directory = "fasta"

    for fasta_file in os.listdir(fasta_directory):
        output_file = os.path.join(fasta_directory, fasta_file.replace(".fasta", "_go.csv"))

        # Check if output file exists and read it if yes
        if os.path.exists(output_file):
            scraped_df = pd.read_csv(output_file)
            processed_ids = set(scraped_df['Protein ID'].tolist())
        else:
            scraped_df = pd.DataFrame(columns=['Protein ID', 'GO ID', 'Function', 'Link'])
            processed_ids = set()

        protein_ids = extract_protein_ids(fasta_file)
        new_entries = []

        for protein in protein_ids:
            if protein in processed_ids:
                continue  # Skip already processed proteins

            link = f"https://www.uniprot.org/uniprotkb/{protein}"
            ids, funcs = get_ids_names(protein)
            new_entry = {
                'Protein ID': protein,
                'GO ID': ids,
                'Function': funcs,
                'Link': link
            }
            new_entries.append(new_entry)
            # Print progress
            print(protein, ids, funcs, link)
            # Add to DataFrame and save immediately
            scraped_df = pd.concat([scraped_df, pd.DataFrame([new_entry])], ignore_index=True)
            scraped_df.to_csv(output_file, index=False)

        print('Processing complete')

if __name__ == "__main__":
    main()
