import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
tqdm.pandas()

if __name__ == "__main__":
    
    # Path to the CycPept database file
    database_file = '../../docs/data/cycpeptdb.csv'

    # Read the database file into a dataframe
    df = pd.read_csv(database_file)

    #Data cleaning operations

    #Remove duplicate rows
    df_cleaned = df.drop_duplicates()

    #Remove rows with duplicate peptide sequences
    df_cleaned = df_cleaned.drop_duplicates(subset='Structurally_Unique_ID')

    # Remove the features (columns) that are irrelevant from the dataframe
    df_cleaned = df_cleaned.drop(['Original_Name_in_Source_Literature', 'Same_Peptides_ID', 'Year', 'Source','Same_Peptides_Source','Same_Peptides_Permeability','Same_Peptides_Assay','Detection_Limit_1','Detection_Limit_2','R_PAMAP','R_MDCK','T_PAMPA','NULL'], axis=1)

    #Save the cleaned dataframe to a new file
    df_cleaned.to_csv('../../docs/data/cycpeptdb_clean.csv', index=False)

    #Print message when done
    print('Data cleaning operations completed successfully!')
