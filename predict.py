from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
from time import time
import random

def get_image():
    dataset_location = "MNIST_data"

    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                ])

    # Download and load the training data
    valset = datasets.MNIST(dataset_location, download=True, train=False, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    images, labels = next(iter(valloader))

    random_image = random.randint(0, len(images))

    return images[random_image]

def load_model(PATH):
    # Save
    return torch.load(PATH)

def validate_model(model, image):
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(image)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))

def main():
    model_path = "my_model.pth"

    image = get_image()

    plt.imshow(image.reshape(28,28), cmap="gray")
    plt.show()

    model = torch.load(model_path)

    validate_model(model, image.view(1, 784))

if __name__ == "__main__":
    main()