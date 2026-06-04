# Proteome FASTA files

These UniProt reference proteome FASTA files are **inputs to `scraper.py`**, which
builds the `multi_proteome_go.duckdb` GO-annotation database. They are **not needed
to run the app** and are intentionally **not tracked in git** (each is 1.8–2.4 MB),
to keep the repository small.

Download the proteomes you need from UniProt into this directory before running the
scraper. The proteome IDs are encoded in the original filenames:

| UniProt Proteome | Organism (taxid) |
|------------------|------------------|
| `UP000000625`    | *Escherichia coli* K-12 (83333) |
| `UP000078142`    | see UniProt |
| `UP000184216`    | see UniProt |
| `UP000565286`    | see UniProt |

Example download (FASTA, canonical sequences):

```bash
# Replace UPXXXXXXXXX with the proteome ID you need
curl -L -o "UP000000625.fasta" \
  "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=proteome:UP000000625"
```

Then build the database:

```bash
pip install ".[scraper]"
python scraper.py
```
