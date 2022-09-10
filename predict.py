import torch
from torchvision import transforms
from PIL import Image

# Specify a path
PATH = "my_model.pt"
image_path = "test/test1.PNG"

# Load
model = torch.load(PATH)
model.eval()

img = Image.open(image_path)

convert_tensor = transforms.ToTensor()

img = convert_tensor(img)
print(img)

# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Specify a path
PATH = "my_model.pt"

# Save
torch.save(model, PATH)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))