
# Train an autopilot for autonomous ground vehicle using
# convolutional neural network and labeled images. 

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import cnn_network
import cv2 as cv

num_parameters = 4 #first parameter is the script name

#Pass in command line arguments for path name 
if len(sys.argv) != num_parameters:
    print(f'Python script needs {num_parameters} parameters!!!')
    sys.exit(1) #Exit with an error code
else:
    data_dir = sys.argv[1]
    model_name = sys.argv[2]
    figure_name = sys.argv[3]
    
model_path = "/home/flashfire/FlashFire/models/"

# Designate processing unit for CNN training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

class CustomImageDataset(Dataset): 

    # Create a dataset from our collected data

    def __init__(self, annotations_file, img_dir, transform=transforms.ToTensor()):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv.imread(img_path, cv.IMREAD_COLOR)
        steering = self.img_labels.iloc[idx, 1].astype(np.float32)
        throttle = self.img_labels.iloc[idx, 2].astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return image.float(), steering, throttle



def train(dataloader, model, loss_fn, optimizer):
    
    # Define Training Function
    
    size = len(dataloader.dataset)
    model.train()
    epoch_loss = 0.0

    for batch, (image, steering, throttle) in enumerate(dataloader):
        # Combine steering and throttle into one tensor (2 columns, X rows)
        target = torch.stack((steering, throttle), -1) 
        X, y = image.to(DEVICE), target.to(DEVICE)

        # Compute prediction error
        pred = model(X)  # forward propagation
        batch_loss = loss_fn(pred, y)  # compute loss
        optimizer.zero_grad()  # zero previous gradient
        batch_loss.backward()  # back propagatin
        optimizer.step()  # update parameters
        
        batch_loss, sample_count = batch_loss.item(), (batch + 1) * len(X)
        epoch_loss = (epoch_loss*batch + batch_loss) / (batch + 1)
        print(f"loss: {batch_loss:>7f} [{sample_count:>5d}/{size:>5d}]")
        
    return epoch_loss

        

def test(dataloader, model, loss_fn):
    
    # Define a test function to evaluate model performance

    #size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for image, steering, throttle in dataloader:
            #Combine steering and throttle into one tensor (2 columns, X rows)
            target = torch.stack((steering, throttle), -1) 
            X, y = image.to(DEVICE), target.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: {test_loss:>8f} \n")

    return test_loss


if __name__ == '__main__':

    #Apply data augmentation to make a more robust training set
    transform = transforms.Compose([
        transforms.RandomResizedCrop(200),
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(30),      # Randomly rotate the image by up to 30 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly adjust brightness, contrast, saturation, and hue
        transforms.RandomGrayscale(p=0.2),  # Randomly convert the image to grayscale with a probability of 0.2
    ])

    # Create a dataset
    annotations_file = data_dir + '/labels.csv'  # the name of the csv file
    img_dir = data_dir + '/images' # the name of the folder with all the images in it
    collected_data = CustomImageDataset(annotations_file, img_dir)
    print("data length: ", len(collected_data))

    # Define the size for train and test data
    train_data_len = len(collected_data)
    train_data_size = round(train_data_len*0.9)
    test_data_size = train_data_len - train_data_size 
    print("len and train and test: ", train_data_len, " ", train_data_size, " ", test_data_size)

    # Load the datset (split into train and test)
    train_data, test_data = random_split(collected_data, [train_data_size, test_data_size])
    train_dataloader = DataLoader(train_data, batch_size=125)
    test_dataloader = DataLoader(test_data, batch_size=125)


    # Initialize the model
    # Models that train well:
    #     lr = 0.001, epochs = 10
    #     lr = 0.0001, epochs = 15 (epochs = 20 might also work)
    
    # Define an optimizer and learning rate scheduler
    lr = 0.001
    model = cnn_network.DonkeyNet().to(DEVICE)# choose the architecture class from cnn_network.py
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = StepLR(optimizer, step_size=5, gamma=0.05)  # Adjust the step_size and gamma as needed
    loss_fn = nn.MSELoss()
    epochs = 15

    # Optimize the model
    train_loss = []
    test_loss = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        training_loss = train(train_dataloader, model, loss_fn, optimizer)
        testing_loss = test(test_dataloader, model, loss_fn)
        print("average training loss: ", training_loss)
        print("average testing loss: ", testing_loss)
        # Apply the learning rate scheduler after each epoch
        #scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate after scheduler step: {current_lr}")
        # save values
        train_loss.append(training_loss)
        test_loss.append(testing_loss)   

    print(f"Optimize Done!")


    #print("final test lost: ", test_loss[-1])
    len_train_loss = len(train_loss)
    len_test_loss = len(test_loss)
    print("Train loss length: ", len_train_loss)
    print("Test loss length: ", len_test_loss)


    # create array for x values for plotting train
    epochs_array = list(range(epochs))

    # Graph the test and train data
    plot_title = f'{model._get_name()} - {epochs} pochs - {lr} learning rate'
    fig = plt.figure()
    axs = fig.add_subplot(1,1,1)
    plt.plot(epochs_array, train_loss, color='b', label="Training Loss")
    plt.plot(epochs_array, test_loss, '--', color='orange', label='Testing Loss')
    axs.set_ylabel('Loss')
    axs.set_xlabel('Training Epoch')
    axs.set_title('Analyzing Training and Testing Loss')
    axs.legend()
    fig.savefig(model_path + figure_name)

    # Save the model
    torch.save(model.state_dict(), model_path + model_name)


    
