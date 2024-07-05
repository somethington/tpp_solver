# TPP Analysis App

<p align="center">
  <img src="logo.svg" alt="Lab Logo" width="200"/>
</p>

This hosted Streamlit application performs Thermal Proteome Profiling (TPP) analysis on protein data.

## Description

The TPP Analysis App processes raw data from TSV files and metadata from CSV files to analyze protein thermal stability. It performs curve fitting, generates plots for each protein, and provides downloadable results.

## Features

- Upload and process TSV (raw data) and CSV (metadata) files
- Filter data based on a user-defined maximum number of allowed zero values
- Impute missing data
- Perform curve fitting using a sigmoid function
- Generate and display plots for each protein
- Provide downloadable ZIP file containing SVG plots and a summary CSV

## Usage

1. Access the TPP Analysis App at [insert hosted app URL here]

2. Upload your TSV (raw data) and CSV (metadata) files using the file upload widgets

3. Adjust the "Maximum number of zeros allowed" as needed

4. Click "Continue Analysis" to process the data and generate results

5. Download the zipped SVG files containing the protein curves and summary data

## Input File Format

### Metadata CSV File

The metadata CSV file should follow this format:

| filename    | Temperature | Treatment | Samples         |
|-------------|-------------|-----------|-----------------|
| C_1_1.mzML  | 45.4        | control   | C_1_1 Intensity |
| C_2_1.mzML  | 45.4        | control   | C_2_1 Intensity |
| C_1_2.mzML  | 49.1        | control   | C_1_2 Intensity |
| C_2_2.mzML  | 49.1        | control   | C_2_2 Intensity |
| ...         | ...         | ...       | ...             |
| T_2_8.mzML  | 80.1        | ADP       | T_2_8 Intensity |
| T_3_8.mzML  | 80.1        | ADP       | T_3_8 Intensity |

Note: There are no specific naming requirements for the metadata CSV file itself. You can name it as you prefer, as long as it's a valid CSV file.

Explanation of the metadata file columns:

1. **filename**: This should be the name of the experimental output file, such as mzML
2. **Temperature**: The temperature at which the sample was measured (in °C).
3. **Treatment**: The treatment condition (e.g., 'control' or 'ADP').
4. **Samples**: The unique identifier for each sample, such as C_1_1 or T_3_2 in the example

Each row represents a unique combination of temperature, treatment, and sample. The 'Intensity' in the Samples column refers to the measurement type in your raw data file.

Ensure your metadata CSV file follows this format for the application to process it correctly.

## Requirements

- TSV file containing intensity data acquired from FragPipe
- CSV file containing metadata (formatted as described above)

## Contact

[Lukas Dolidze] - [ldoli002@ucr.edu]

App URL: [insert hosted app URL here]