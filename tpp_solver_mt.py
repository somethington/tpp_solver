'''
 ,`````.          _________
' AWAU  `,       /_  ___   \
'        `.     /@ \/@  \   \
 ` , . , '  `.. \__/\___/   /
                 \_\/______/
                 /     /\\\\\
                |     |\\\\\\
                 \      \\\\\\
                  \______/\\\\     -ccw-
            _______ ||_||_______
           (______(((_(((______(@)
           Cooked, Fried, and Prepared by Mr Gugi
'''
import pandas as pd


# File reading functions
def read_tsv_file(file_path):
    return pd.read_csv(file_path, sep='\t')

def read_csv_file(file_path):
    return pd.read_csv(file_path)

def main():
    pass

if __name__ == "__main__":
    main()