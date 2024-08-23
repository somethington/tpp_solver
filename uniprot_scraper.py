from Bio import SeqIO
import pandas as pd
import requests

fasta_file = "UP000000625_83333.fasta"

def get_ids_names(protein):
    url = f"https://rest.uniprot.org/uniprotkb/{protein}?fields=go_id"
    headers = {
        "Accept": "text/plain; format=tsv"
    }

    response = requests.get(url, headers=headers)
    res_list = [s.replace(";", "") for s in ((response.text).split()[3:])]

    if not res_list or len(res_list) == 0:
        return "NA", "NA"

    url = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/" + ",".join([s.replace(":", "%3A") for s in res_list])
    headers = {
        "Accept": "application/json"
    }

    response = requests.get(url, headers)
    data = response.json()

    ids = ""
    funcs = ""

    for item in data['results']:
        ids += item['id'] + "|"
        funcs += item['name'] + "|"

    ids = ids[:-1]
    funcs = funcs[:-1]

    return ids, funcs

def extract_protein_ids(fasta_file):
    protein_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        description = record.description
        protein_id = description.split('|')[1]
        protein_ids.append(protein_id)
    return protein_ids

def main():
    # Initialize dataframe with protein IDs
    scraped_df = pd.DataFrame(columns=['Protein ID', 'GO ID', 'Function', 'Link'])
    protein_ids = extract_protein_ids(fasta_file)

    rows = []
    for protein in protein_ids:
        link = f"https://www.uniprot.org/uniprotkb/{protein}"
        ids, funcs = get_ids_names(protein)
        # Collect new row data
        rows.append({
            'Protein ID': protein,
            'GO ID': ids,
            'Function': funcs,
            'Link': link
        })
    # Append all rows to the dataframe at once
    scraped_df = pd.concat([scraped_df, pd.DataFrame(rows)], ignore_index=True)

    # Save dataframe to CSV
    scraped_df.to_csv("parsed_output.csv", index=False)    
    print('done')
if __name__ == "__main__":
    main()
