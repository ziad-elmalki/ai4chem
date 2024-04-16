import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import pickle 
from rdkit import Chem
from rdkit.Chem import Draw


# Read the cycpepdb_clean as a dataframe
df = pd.read_csv('../../docs/data/cycpeptdb_clean.csv')

PepNet_data = pd.DataFrame(columns=['Permeability', 'Image'])

# Create a new column 'images' to store the RDKit images of the peptides
PepNet_data['Image'] = df['SMILES'].apply(lambda x: Draw.MolToImage(Chem.MolFromSmiles(x)))

#Create column Permeability and store the permeability values
PepNet_data['Permeability'] = df['Permeability']

# Save the data to a Pickle file
PepNet_data.to_pickle('../../docs/data/PepNet_data.pkl')

# Load the data from the Pickle file
PepNet_data = pd.read_pickle('../../docs/data/PepNet_data.pkl')

# Visualize the first few samples from the DataFrame
num_samples_to_visualize = 5

for i in range(num_samples_to_visualize):
    # Get the image data
    image_data = PepNet_data.loc[i, 'Image']
    
    # Get the permeability value
    permeability_value = PepNet_data.loc[i, 'Permeability']
    
    # Visualize the image and permeability value
    plt.figure(figsize=(4, 4))
    plt.imshow(image_data)
    plt.title(f'Permeability Value: {permeability_value}')
    plt.axis('off')
    plt.show()