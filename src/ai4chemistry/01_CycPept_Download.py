'This is a script to download the CycPeptDB database from the CycPeptMPDB website.'

import os
import urllib.request


if __name__ == "__main__":
    
    # URL of the CycPeptDB database
    url = "http://cycpeptmpdb.com/static//download/peptides/CycPeptMPDB_Peptide_All.csv"

    # Destination directory
    destination_dir = "../../docs/data"

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Download the database file
    urllib.request.urlretrieve(url, os.path.join(destination_dir, "cycpeptdb.csv"))

    print("CycPeptDB downloaded is complete, you can find it stored in docs/data directory.")