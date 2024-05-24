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

from sklearn.model_selection import train_test_split

from torchvision import transforms
import h5py

device = "cuda" if torch.cuda.is_available() else "cpu"
print("GPU available: ", torch.cuda.is_available())

np.random.seed(69)

# Define functions 
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
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    running_loss = 0
    model.train()  # Set the model to training mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)  # Pass the data to the model to execute the model forward
        loss = loss_fn(pred, y)
        running_loss += loss.item()

        # Backpropagation
        loss.backward()  # Compute gradients of the loss w.r.t parameters (backward pass)
        optimizer.step()  # Do a gradient descent step and adjust parameters
        optimizer.zero_grad()  # Reset the gradients of model parameters to zero (gradients by default add up)

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    running_loss /= num_batches
    return running_loss

# Define the test function
def test(dataloader, model, loss_fn, best_loss):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    
    with torch.no_grad():  # Do not track gradients while evaluating (faster)
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # Forward pass
            pred = model(X)
            pred = pred.squeeze(1)
            # Compute loss
            batch_loss = loss_fn(pred, y)
            test_loss += batch_loss.item()
            for pred_i, y_i in zip(pred, y):
                if torch.abs(pred_i - y_i) < 0.1 * torch.abs(y_i):
                    correct += 1

    # Average loss over all batches
    test_loss /= num_batches
    correct /= size

    # Save the model if the test loss is the lowest
    if test_loss <= best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), './best_model.pth')
        print(f'Saving model with validation loss {best_loss:.4f}')

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, best_loss


# Grid search function
def grid_search(hyperparameters, train_main_dataloader, test_main_dataloader):
    results = []
    best_loss = float('inf')
    best_params = None

    for params in hyperparameters:
        epochs = params['epochs']
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']

        print(f"Training with params: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}")

        cnn_model = nn.Sequential( 
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(73984, 4096),  # 256 * 6 * 6
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1)
        ).to(device)

        loss_fn = nn.MSELoss()
        optimizer = Adam(cnn_model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss = train(train_main_dataloader, cnn_model, loss_fn, optimizer)
            test_loss, best_loss = test(test_main_dataloader, cnn_model, loss_fn, best_loss)

        results.append({
            'params': params,
            'best_loss': best_loss
        })

        if best_loss < best_loss:
            best_params = params
            best_loss = test_loss

    return best_params, results


if __name__ == "__main__":
    
    # Load the data from the h5py file
    h5file = '../../docs/data/Augmented_PepNet_data.h5'

    with h5py.File(h5file, 'r') as F:
        images = np.array(F['images'])
        labels = np.array(F['permeability'])

    # Create numpy arrays for labels and data
    data = torch.from_numpy(images)
    print(data.shape)
    labels = torch.from_numpy(labels)
    print(labels.shape)

    # Change the data type and permute the color channel axis
    data = data.type(torch.float32).permute(0, 3, 1, 2)  # leave this as is
    labels = labels.type(torch.float32)  # leave this as is
    print(data.shape, labels.shape)

    # Select a random subset of images for computing mean and std
    subset_indices = torch.randint(0, len(data), (1000,))
    subset_images = data[subset_indices]

    # Compute mean and std from the entire dataset
    mean = torch.mean(subset_images .float(), dim=(0, 2, 3))
    std = torch.std(subset_images.float(), dim=(0, 2, 3))

    print("Computed Mean:", mean)
    print("Computed Std:", std)

    # Define the composed transform without cropping
    composed_transform = transforms.Compose([
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    # Apply the transform to the entire dataset to see if it worked 
    Test_Normalization_data = composed_transform(data)

    # Now, transformed_images contains the normalized images
    # You can check the mean and std of the transformed images
    print("Transformed Mean:", torch.mean(Test_Normalization_data.float(), dim=(0, 2, 3)))
    print("Transformed Std:", torch.std(Test_Normalization_data.float(), dim=(0, 2, 3)))

    # Split the data into training and test sets
    train_main_idx, test_main_idx = train_test_split(np.arange(labels.shape[0]), train_size=13000)
    train_main_images, train_main_labels, test_main_images, test_main_labels = data[train_main_idx], labels[train_main_idx], data[test_main_idx], labels[test_main_idx]

    # Create the datasets 
    Train_main_dataset = MyDataset(train_main_images, train_main_labels, transform=composed_transform)
    Test_main_dataset = MyDataset(test_main_images, test_main_labels, transform=composed_transform)

    
    # Define hyperparameters for grid search
    hyperparameters = [
        {'epochs': 20, 'batch_size': 32, 'learning_rate': 1e-3},
        {'epochs': 20, 'batch_size': 64, 'learning_rate': 1e-3},
        {'epochs': 30, 'batch_size': 32, 'learning_rate': 1e-4},
        {'epochs': 30, 'batch_size': 64, 'learning_rate': 1e-4}
    ]

    # Define the architecture (AlexNet)
    cnn_model = nn.Sequential( 
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(64, 192, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(192, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        nn.Linear(73984, 4096),  # 256 * 6 * 6
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 1)
    ).to(device)

    loss_fn = nn.MSELoss()
    best_loss = float('inf')
    best_params = None
    results_grid_search = []
    for params in hyperparameters:

        nb_epochs = params['epochs']
        bs = params['batch_size']
        lr = params['learning_rate']

        print(f"Training with params: epochs={nb_epochs}, batch_size={bs}, learning_rate={lr}")

        optimizer_full_cnn = torch.optim.Adam(cnn_model.parameters(), lr) # Pass model parameters to optimizer
        
        train_dataloader = DataLoader(Train_main_dataset, batch_size=bs, shuffle=True)
        test_dataloader = DataLoader(Test_main_dataset, batch_size=bs)
        # Perform grid search
        #best_params, results = grid_search(cnn_model, hyperparameters, train_dataloader, test_dataloader)

        #print(f"Best hyperparameters: {best_params}")
        #print(f"Results: {results}")

        # Train the model
        Losses_train_convo = []
        Losses_test_convo = []
        
        for t in range(nb_epochs):
            print(f"Epoch {t+1}\n")

            train_loss = train(train_dataloader, cnn_model, loss_fn, optimizer_full_cnn)
            Losses_train_convo.append(train_loss)

            test_loss, best_loss = test(test_dataloader, cnn_model, loss_fn, best_loss)
            Losses_test_convo.append(test_loss)

            best_loss = best_loss
        
        
        results_grid_search.append({
            'params': params,
            'best_loss': best_loss
        })

        if best_loss < best_loss:
            best_params = params
            #best_loss = test_loss

    print(" Done ! \n Grid search results: ", results_grid_search)
    print("Best hyperparameters: ", best_params)
    
    
    # Load the best model
    best_model = cnn_model
    best_model.load_state_dict(torch.load('./best_model.pth'))

    # Evaluate the model on the test set
    best_model.eval()
    true_list = []
    pred_list = []
    total_correct = 0
    total_samples = 0
    mse =0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = best_model(images)
            outputs = outputs.squeeze(1)
            true_list.extend(labels.tolist())
            pred_list.extend(outputs.tolist())

            #Compute the mean squared error
            batch_mse = torch.mean((outputs - labels)**2).item()
            mse += batch_mse

            # Compute the number of correct predictions within a threshold of 10% of the true value
            for output, label in zip(outputs, labels):
                if torch.abs(output - label) < 0.1 * torch.abs(label):
                    total_correct += 1
            total_samples += labels.size(0)

    # Compute the mean squared error
    mse /= len(test_dataloader)

    # Compute the accuracy
    accuracy = total_correct / total_samples * 100
    print(f"Test accuracy: {accuracy:.2f}%")

    # Save the predictions to a csv file
    df = pd.DataFrame({"True Values": true_list, "Predicted Values": pred_list})
    df.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")
