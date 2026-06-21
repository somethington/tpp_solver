"""File reading and figure serialization helpers."""
import io

import matplotlib.pyplot as plt
import pandas as pd

def read_tsv_file(file_path):
    return pd.read_csv(file_path, sep='\t')

def read_csv_file(file_path):
    return pd.read_csv(file_path)

def save_figure_to_svg(name, fig):
    svg_io = io.BytesIO()
    fig.savefig(svg_io, format='svg', bbox_inches='tight')
    svg_io.seek(0)
    plt.close(fig)  # Ensure figure is closed immediately
    return name, svg_io.getvalue()
