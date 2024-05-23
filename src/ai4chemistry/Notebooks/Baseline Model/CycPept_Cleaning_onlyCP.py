import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
tqdm.pandas()


df = pd.read_csv('../../../../docs/data/cycpeptdb.csv')

#Data cleaning operations

#Remove duplicate rows
df_cleaned = df.drop_duplicates()

#Remove rows with duplicate peptide sequences
df_cleaned = df_cleaned.drop_duplicates(subset='Structurally_Unique_ID')

#Specify the columns you want to keep
columns_to_keep = ['SMILES', 'PC1', 'PC2', 'Permeability']

# Keep only the specified columns
df_cleaned = df_cleaned[columns_to_keep]

# Save the resulting DataFrame to a new CSV file
df_cleaned.to_csv('../../../../docs/data/cycpeptdb_clean_onlyCP.csv', index=False)

#Print message when done
print('Data cleaning operations completed successfully!')