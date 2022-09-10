# Import necessary packages
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
    trainset = datasets.MNIST(dataset_location, download=True, train=True, transform=transform)
    valset = datasets.MNIST(dataset_location, download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    return trainloader, valloader

def build_model(images, labels, criterion):
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

    logps = model(images)
    loss = criterion(logps, labels)

    loss.backward()

    return model

def optimize_model(model, trainloader, criterion, images):
    # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    images, labels = next(iter(trainloader))
    images.resize_(64, 784)

    # Clear the gradients, do this because gradients are accumulated
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()

    # Take an update step and few the new weights
    optimizer.step()

    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    return optimizer

def train_model(model, trainloader, criterion, optimizer, epochs):
    starting_time = time()
    print("Starting training model")
    for e in range(epochs):
        running_loss = 0
        e_time = time()
        for images, labels in trainloader:
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
            print("Epoch {} : {}s - Training loss: {}".format(e + 1, round(time() - e_time, 1), running_loss/len(trainloader)))
    print("\nTraining Time = {} min".format(round((time()-starting_time)/60, 2)))

def view_classify(img, ps):
    ''' 
        Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

def save_model(model, PATH):
    # Save
    torch.save(model, PATH)

def validate_model(model, valloader):
    images, labels = next(iter(valloader))

    random_image = random.randint(0, len(images))
    img = images[random_image].view(1, 784)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))
    view_classify(img.view(1, 28, 28), ps)

def main():
    trainloader, valloader = prepare_dataset()
    criterion = nn.NLLLoss()
    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)
    epochs = 20

    model = build_model(images, labels, criterion)

    optimizer = optimize_model(model, trainloader, criterion, images)

    train_model(model, trainloader, criterion, optimizer, epochs)

    validate_model(model, valloader)

if __name__ == "__main__":
    main()