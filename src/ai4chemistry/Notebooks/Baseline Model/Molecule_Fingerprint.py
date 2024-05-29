from rdkit import Chem
import tmap as tm
from map4 import MAP4Calculator
import pandas as pd

dim = 1024

MAP4 = MAP4Calculator(dimensions=dim)
ENC = tm.Minhash(dim)

df = pd.read_csv('../../../../docs/data/cycpeptdb_clean.csv')

df['MolFromSmiles'] = df['SMILES'].apply(Chem.MolFromSmiles)
df['fingerprint'] = MAP4.calculate_many(df['MolFromSmiles'])

df.to_csv('../../../../docs/data/cycpeptdb_clean_fps.csv')

#creation of a code to add the 'fingerprint' feature to the database cycpepcycpeptdb_clean_fps.csv