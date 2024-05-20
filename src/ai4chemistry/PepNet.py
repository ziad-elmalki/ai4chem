import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from torchvision import transforms
import h5py
np.random.seed(69)


#Define functions 

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None, target_transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)

        y = self.target[index]
        if self.target_transform:
            y = self.target_transform(y)
        
    

        return x, y
    

# Define the training function
def train(dataloader, model, loss_fn, optimizer):
    '''
    This function implements the train loop. It iterates over the training dataset
    and try to converge to optimal parameters.
    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    running_loss = 0
    model.train() # Set the model to training mode
    for batch, (X, y) in enumerate(dataloader):
        
        # Compute prediction and loss
        pred = model(X) # Pass the data to the model to execute the model forward
        loss = loss_fn(pred, y)
        running_loss += loss.item()

        # Backpropagation
        loss.backward() # Compute gradients of the loss w.r.t parameters (backward pass)
        optimizer.step() # Do a gradient descent step and adjust parameters
        optimizer.zero_grad() # Reset the gradients of model parameters to zero (gradients by default add up)

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    running_loss /= num_batches
    
    return running_loss

# Define the test function
# def test(dataloader, model, loss_fn):
#     '''
#     This function implements the validation/test loop. It iterates over the test
#     dataset to check if the model performance is improving.
#     '''
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval() # Set the model to evaluation mode
#     test_loss, correct = 0, 0
#     with torch.no_grad(): # Do not track gradients while evaluating (faster)
#         for X, y in dataloader:
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item() # Compute CE loss on the batch
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item() # Compute classification error
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#     return test_loss

def test(dataloader, model, loss_fn):
    '''
    This function implements the validation/test loop. It iterates over the test
    dataset to check if the model performance is improving.
    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # Do not track gradients while evaluating (faster)
        for X, y in dataloader:
            # Forward pass
            pred = model(X)
            pred = pred.squeeze(1)
            #print(pred,y)
            # Compute loss
            batch_loss = loss_fn(pred, y)
            test_loss += batch_loss.item()
            for pred_i, y_i in zip(pred, y):
                if torch.abs(pred_i - y_i) < 0.1 * torch.abs(y_i):
                    correct += 1

    # Average loss over all batches
    test_loss /= num_batches
    correct /= size
    # Print test loss
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss


if __name__ == "__main__":
    
    # Load the data from the h5py file
    h5file = '../../docs/data/Augmented_PepNet_data.h5'

    with h5py.File(h5file, 'r') as F:
        #print(type(F['images'][0]))
        images = np.array(F['images'])
        #print(type(images[0]))
        labels = np.array(F['permeability'])

    #Verify the data
    #plt.imshow(images[0])
    #plt.show()
    #print(labels[0])
    
    # create numpy arrays for labels and data
    data = torch.from_numpy(images)
    print(data.shape)
    labels = torch.from_numpy(labels)
    
    # Compute the mean and std of the labels
    labels_mean = torch.mean(labels)
    labels_std = torch.std(labels)

    # Standardize the labels
    labels = (labels - labels_mean) / labels_std
    print(labels.shape)
    

    # we change the data type and permute the color channel axis from place 3 to 1, to conform with pytorch defaults.
    data = data.type(torch.float32).permute(0,3,1,2)  # leave this as is
    labels = labels.type(torch.float32)            # leave this as is
    print(data.shape, labels.shape)

    # continue with computing the channel means and std's after cropping on a subset of the data

    # Compute mean and std from a random crop
    random_crop_transform = transforms.RandomCrop(size=(256, 256))

    # Select a random subset of images for computing mean and std
    subset_indices = torch.randint(0, len(data), (1000,))
    subset_images = data[subset_indices]

    # Apply random crop to the subset
    subset_cropped_images = random_crop_transform(subset_images)

    # Compute mean and std for each channel
    mean = torch.mean(subset_cropped_images.float(), dim=(0, 2, 3))
    std = torch.std(subset_cropped_images.float(), dim=(0, 2, 3))
    #mean = torch.mean(subset_images.float(), dim=(0, 2, 3))
    #std = torch.std(subset_images.float(), dim=(0, 2, 3))

    print("Computed Mean:", mean)
    print("Computed Std:", std)


    # define the composed transform
    composed_transform = transforms.Compose([
        #transforms.RandomCrop(size=(48, 48)),
        transforms.RandomCrop(size=(256, 256)),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    # Apply the transform to the entire dataset to see if it worked 
    Test_Normalization_data = composed_transform(data)

    # Now, transformed_images contains the normalized images
    # You can check the mean and std of the transformed images
    print("Transformed Mean:", torch.mean(Test_Normalization_data.float(), dim=(0, 2, 3)))
    print("Transformed Std:", torch.std(Test_Normalization_data.float(), dim=(0, 2, 3)))

    # Split the data into training and test sets
    train_main_idx, test_main_idx = train_test_split(np.arange(labels.shape[0]), train_size=12000)
    train_main_images, train_main_labels, test_main_images, test_main_labels = data[train_main_idx], labels[train_main_idx], data[test_main_idx], labels[test_main_idx]

    # Create the datasets 
    Train_main_dataset = MyDataset(train_main_images, train_main_labels, transform=composed_transform)
    Test_main_dataset = MyDataset(test_main_images,test_main_labels, transform=composed_transform)

    # Create data loaders with batch size 64
    batch_size = 64

    # Create data loaders for the main data set.
    train_main_dataloader = DataLoader(Train_main_dataset, batch_size=batch_size, shuffle=True)
    test_main_dataloader = DataLoader(Test_main_dataset, batch_size=batch_size)

    # Create the First draft of the model
    cnn_model = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=5),    # Input: 256x256 -> Output: 252x252
    nn.ReLU(),                        
    nn.MaxPool2d(2, 2),               # Output: 252x252 -> 126x126

    nn.Conv2d(6, 16, kernel_size=5),   # Output: 126x126 -> 122x122
    nn.ReLU(),                        
    nn.MaxPool2d(2, 2),               # Output: 122x122 -> 61x61

    nn.Flatten(),                     # Flatten: 16*61*61

    nn.Linear(16*61*61, 120),         # 16*61*61 = 59536
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 1)                  # Single output for regression
    )
    

    # Define the model architecture
    #cnn_model = nn.Sequential(
    #nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    #nn.ReLU(),
    #nn.MaxPool2d(kernel_size=2, stride=2),

    #nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    #nn.ReLU(),
    #nn.MaxPool2d(kernel_size=2, stride=2),

    #nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    #nn.ReLU(),
    #nn.MaxPool2d(kernel_size=2, stride=2),

    #nn.Flatten(),

    #nn.Linear(64*6*6, 256),
    #nn.ReLU(),
    #nn.Dropout(0.5),

    #nn.Linear(256, 128),
    #nn.ReLU(),
    #nn.Dropout(0.5),

    #nn.Linear(128, 64),
    #nn.ReLU(),

    #nn.Linear(64, 32),
    #nn.ReLU(),
    #nn.Dropout(0.5),

    #nn.Linear(32, 1)
    #)

    # Define the loss function and the optimizer
    loss_fn = nn.MSELoss()
    learning_rate_full = 1e-3
    optimizer_full_cnn = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate_full) # Pass model parameters to optimizer

    # Train the model
    Losses_train_convo = []
    Losses_test_convo = []
    epochs = 80
    for t in range(epochs):
        print(f"Epoch {t+1}\n")
        Losses_train_convo.append(train(train_main_dataloader, cnn_model, loss_fn, optimizer_full_cnn))
        Losses_test_convo.append(test(test_main_dataloader, cnn_model, loss_fn))

    print("Done!")

    # Plot the train and test losses
    plt.plot(range(epochs),Losses_train_convo)

    plt.plot(range(epochs),Losses_test_convo)
    plt.legend(["Train Loss","Test Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("ConvoNet: Train and Test Loss as a function of epochs")
    plt.show()

    correct = 0
    total = 0
    mse = 0
    threshold = 0.1 # Threshold for the prediction to be correct
    with torch.no_grad():

        for data in test_main_dataloader:

            images, labels = data
            
            predicted = cnn_model(images)
            predicted = predicted.squeeze(1)
            print("Predicted values : ", predicted*labels_std + labels_mean)
            print("Actual values : ", labels*labels_std + labels_mean)
    
            total += labels.size(0)
    
            for pred_i, y_i in zip(predicted, labels):
                if torch.abs(pred_i - y_i) < threshold * torch.abs(y_i):
                    correct += 1

            batch_mse = torch.mean((predicted - labels)**2).item()
            mse += batch_mse

        mse /= len(test_main_dataloader)
        accuracy = 100 * correct / total

    #print(f'Test accuracy: {100 * correct / total}')
    print(f'Mean Squared Error (MSE): {mse:.6f}')
    print(f'Accuracy within {threshold * 100}% error: {accuracy:.2f}%')