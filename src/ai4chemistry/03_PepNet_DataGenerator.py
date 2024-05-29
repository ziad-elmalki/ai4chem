'This is a script to generate the PepNet dataset from the CycPeptDB database file.'

import numpy as np 
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import h5py

if __name__ == '__main__':
    # Read the cycpepdb_clean as a dataframe
    df = pd.read_csv('../../docs/data/cycpeptdb_clean.csv')

    # Drop rows where permeability is -10
    df = df[df['Permeability'] != -10]
    df.reset_index(drop=True, inplace=True)

    PepNet_data = pd.DataFrame(columns=['Permeability', 'Image'])
    
    # Create a new column 'images' to store the RDKit images of the peptides
    PepNet_data['Image'] = df['SMILES'].apply(lambda x: Draw.MolToImage(Chem.MolFromSmiles(x)))

    #Create column Permeability and store the permeability values
    PepNet_data['Permeability'] = df['Permeability']

    # Define the shape of the image dataset
    nfiles = len(PepNet_data)
    IMG_WIDTH = PepNet_data['Image'][0].size[0]
    IMG_HEIGHT = PepNet_data['Image'][0].size[1]
    
    # Create an h5 file
    h5file = '../../docs/data/PepNet_data.h5'

    with h5py.File(h5file, 'w') as h5f:
        # Create the image dataset
        img_ds = h5f.create_dataset('images', shape=(nfiles, IMG_WIDTH, IMG_HEIGHT, 3), dtype='uint8') #dtype used to be int
        
        # Save the images to the dataset
        for i in range(nfiles):
            img_ds[i] = np.array(PepNet_data['Image'][i])
        
        # Create the permeability dataset
        perm_ds = h5f.create_dataset('permeability', shape=(nfiles,), dtype=float)
        
        # Save the permeability values to the dataset
        perm_ds[:] = PepNet_data['Permeability']
    
    print('PepNet Data saved to', h5file)