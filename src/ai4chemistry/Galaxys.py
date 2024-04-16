import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from torchvision import transforms

np.random.seed(69)


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
    


# First create np.arrays from the data in the file, then make torch.Tensors from those.
# Directly going to torch.Tensor would be very slow in this case because h5py provides lazy access to the data, (see https://github.com/pytorch/pytorch/issues/13918).
with h5py.File('Galaxy10.h5', 'r') as F:
  images = np.array(F['images'])
  labels = np.array(F['ans'])

# create numpy arrays for labels and data
data = torch.from_numpy(images)
labels = torch.from_numpy(labels)

# we change the data type and permute the color channel axis from place 3 to 1, to conform with pytorch defaults.
data = data.type(torch.float32).permute(0,3,1,2)  # leave this as is
labels = labels.type(torch.LongTensor)            # leave this as is

print(data.shape, labels.shape)

# continue with computing the channel means and std's after cropping on a subset of the data

# Compute mean and std from a random crop
random_crop_transform = transforms.RandomCrop(size=(48, 48))

# Select a random subset of images for computing mean and std
subset_indices = torch.randint(0, len(data), (1000,))
subset_images = data[subset_indices]

# Apply random crop to the subset
subset_cropped_images = random_crop_transform(subset_images)

# Compute mean and std for each channel
mean = torch.mean(subset_cropped_images.float(), dim=(0, 2, 3))
std = torch.std(subset_cropped_images.float(), dim=(0, 2, 3))

print("Computed Mean:", mean)
print("Computed Std:", std)


# define the composed transform
composed_transform = transforms.Compose([
    transforms.RandomCrop(size=(48, 48)),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])

# Apply the transform to the entire dataset to see if it worked 
Test_Normalization_data = composed_transform(data)

# Now, transformed_images contains the normalized images
# You can check the mean and std of the transformed images
print("Transformed Mean:", torch.mean(Test_Normalization_data.float(), dim=(0, 2, 3)))
print("Transformed Std:", torch.std(Test_Normalization_data.float(), dim=(0, 2, 3)))

# Identify indices for the main dataset and held-out dataset
main_dataset_indices = labels< 8
transfer_indices = (labels== 8) | (labels== 9)

# Create main dataset
main_dataset_data = data[main_dataset_indices]
main_dataset_labels = labels[main_dataset_indices]

# Create held-out dataset for transfer learning
transfer_dataset_data = data[transfer_indices]
transfer_dataset_labels = labels[transfer_indices] - 8

# Verify the shapes and labels
print("Main Dataset Shape:", main_dataset_data.shape, main_dataset_labels.shape)
print("Held-out Dataset Shape:", transfer_dataset_data.shape, transfer_dataset_labels.shape)
print("Held-out Dataset labels change verification", transfer_dataset_labels[:5])

# create train and test datasets for the two tasks as instances of the MyDataset class, e.g.
# example_dataset = MyDataset(example_data, example_labels, transform=example_transform)

#For the main Data_set (labels 0 to 7)
train_main_idx, test_main_idx = train_test_split(np.arange(main_dataset_labels.shape[0]), train_size=18000)
train_main_images, train_main_labels, test_main_images, test_main_labels = main_dataset_data[train_main_idx], main_dataset_labels[train_main_idx], main_dataset_data[test_main_idx], main_dataset_labels[test_main_idx]

Train_main_dataset = MyDataset(train_main_images, train_main_labels, transform=composed_transform)
Test_main_dataset = MyDataset(test_main_images,test_main_labels, transform=composed_transform)

#For the held out Dataset (labels 8 and 9 that were transformed to 0 and 1)
train_transfer_idx, test_transfer_idx = train_test_split(np.arange(transfer_dataset_labels.shape[0]), train_size=1000)
train_transfer_images, train_transfer_labels, test_transfer_images, test_transfer_labels = transfer_dataset_data[train_transfer_idx], transfer_dataset_labels[train_transfer_idx], transfer_dataset_data[test_transfer_idx], transfer_dataset_labels[test_transfer_idx]

Train_transfer_dataset = MyDataset(train_transfer_images, train_transfer_labels, transform=composed_transform)
Test_transfer_dataset = MyDataset(test_transfer_images,test_transfer_labels, transform=composed_transform)


# Create data loaders with batch size 64
batch_size = 64

# Create data loaders for the main data set.
train_main_dataloader = DataLoader(Train_main_dataset, batch_size=batch_size, shuffle=True)
test_main_dataloader = DataLoader(Test_main_dataset, batch_size=batch_size)

# Create data loaders for the transfer learning.
train_transfer_dataloader = DataLoader(Train_transfer_dataset, batch_size=batch_size, shuffle=True)
test_transfer_dataloader = DataLoader(Test_transfer_dataset, batch_size=batch_size)


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

        if batch % 200 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    running_loss /= num_batches
    
    return running_loss

# Define the test function
def test(dataloader, model, loss_fn):
    '''
    This function implements the validation/test loop. It iterates over the test
    dataset to check if the model performance is improving.
    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # Set the model to evaluation mode
    test_loss, correct = 0, 0
    with torch.no_grad(): # Do not track gradients while evaluating (faster)
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item() # Compute CE loss on the batch
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # Compute classification error
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


# Define the model CNN


# Define and train the CNN
cnn_backbone = nn.Sequential(
    nn.Conv2d(3, 8, 5),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(8, 16, 5),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Flatten(),

    nn.Linear(1296, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU()
)

# Define the classifier head
classifier_head = nn.Linear(64, 8)
    

# Stack the cnn backbone and the classifier head using nn.Sequential
full_cnn_model = nn.Sequential(
    cnn_backbone,
    classifier_head,
)

# Define the loss function and the optimizer
loss_fn = nn.CrossEntropyLoss()
learning_rate_full = 2e-3
optimizer_full_cnn = torch.optim.Adam(full_cnn_model.parameters(), lr=learning_rate_full) # Pass model parameters to optimizer

# Train the model
Losses_train_convo = []
Losses_test_convo = []
epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n")
    Losses_train_convo.append(train(train_main_dataloader, full_cnn_model, loss_fn, optimizer_full_cnn))
    Losses_test_convo.append(test(test_main_dataloader, full_cnn_model, loss_fn))

print("Done!")

# Plot the train and test losses
plt.plot(range(epochs),Losses_train_convo)

plt.plot(range(epochs),Losses_test_convo)
plt.legend(["Train Loss","Test Loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("ConvoNet: Train and Test Loss as a function of epochs")

correct = 0
total = 0

with torch.no_grad():

    for data in test_main_dataloader:

        images, labels = data

        outputs = full_cnn_model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test accuracy: {100 * correct / total}')