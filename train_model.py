# Import pip packages
import numpy as np
import torch
from torch import nn, optim
import torchvision
import matplotlib.pyplot as plt
from time import time
import os
import random

from torchvision import datasets, transforms

def prepare_dataset():
    # Define a transform to normalize the data
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                ])

    dataset_location = "MNIST_data"

    # Download and load the training data
    train_dataset = datasets.MNIST(dataset_location, download=True, train=True, transform=transform)
    validate_dataset = datasets.MNIST(dataset_location, download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=64, shuffle=True)

    return train_loader, validate_loader

def build_model():
    # Layer details for the neural network
    input_size = 784
    hidden_sizes = [64, 16]
    output_size = 10

    # Build a feed-forward network
    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], output_size),
        nn.LogSoftmax(dim=1)
    )

    return model

def optimize_model(model, train_loader, criterion, images):
    # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    images, labels = next(iter(train_loader))
    images.resize_(64, 784)


    # Clear the gradients, do this because gradients are accumulated
    optimizer.zero_grad()

    # Take an update step and few the new weights
    optimizer.step()

    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    return optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
    starting_time = time()
    print("Starting training model")
    for e in range(epochs):
        running_loss = 0
        e_time = time()
        for images, labels in train_loader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
        
            # Training pass
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            
            #This is where the model learns by backpropagating
            loss.backward()
            
            #And optimizes its weights here
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print("Epoch {} : {}s - Training loss: {}".format(e + 1, round(time() - e_time, 1), running_loss/len(train_loader)))
    print("\nTraining Time = {} min".format(round((time()-starting_time)/60, 2)))

def save_model(model, PATH):
    # Save
    torch.save(model, PATH)

def validate_model(model, img):
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(img)

    # Output of the network is the probability
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))

def main():
    train_loader, validate_loader = prepare_dataset()
    criterion = nn.NLLLoss()
    images, labels = next(iter(train_loader))
    epochs = 20

    model = build_model()

    optimizer = optimize_model(model, train_loader, criterion, images)

    train_model(model, train_loader, criterion, optimizer, epochs)

    random_image = random.randint(0, len(images))

    img = images[random_image].view(1, 784)

    validate_model(model, img)

    path = "my_model.pth"

    save_model(model, path)

if __name__ == "__main__":
    main()