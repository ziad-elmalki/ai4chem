'This script performs data augmentation on the PepNet dataset and saves the augmented data to a new HDF5 file.'

import h5py
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# Define the data augmentation transformations
data_transforms = transforms.Compose([
    transforms.ToPILImage(),                  # Convert numpy array to PIL Image
    transforms.RandomHorizontalFlip(),       # Randomly flip the image horizontally
    transforms.RandomVerticalFlip(),         # Randomly flip the image vertically
    transforms.RandomRotation(degrees=15),   # Randomly rotate the image by a maximum of 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust brightness, contrast, saturation, and hue
])

# Path to the HDF5 file
hdf5_file_path = '../../docs/data/PepNet_data.h5'

# Open the HDF5 file in read mode
with h5py.File(hdf5_file_path, 'r') as f:
    images = np.array(f['images'][:])
    labels = np.array(f['permeability'][:])

images = images.astype(np.uint8)


# Create an empty list to store augmented images and labels
augmented_images = []
augmented_labels = []

# Perform data augmentation on each image
for image, label in zip(images, labels):
    # Append original image and label
    augmented_images.append(image)
    augmented_labels.append(label)

    # Apply data augmentation transformations
    augmented_image = data_transforms(image)

    # Convert augmented image to numpy array and append to the list
    augmented_images.append(np.array(augmented_image))

    # Append the label again for the augmented image
    augmented_labels.append(label)


# Save augmented data to a new HDF5 file
output_hdf5_file = '../../docs/data/Augmented_PepNet_data.h5'

with h5py.File(output_hdf5_file, 'w') as f:
    img_ds = f.create_dataset('images', shape=(len(augmented_images), 300, 300, 3), dtype='uint8')
    # Save the images to the dataset
    for i in range(len(augmented_images)):
        img_ds[i] = augmented_images[i]

    perm_ds = f.create_dataset('permeability', shape=(len(augmented_labels),), dtype=float)
    perm_ds[:] = augmented_labels 

print("Data augmentation completed and saved to", output_hdf5_file)

