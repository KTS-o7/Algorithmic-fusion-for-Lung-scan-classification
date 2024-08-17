import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.io import read_image

import dill

from torchvision import models

# Define your get_models function
def get_models(name):
    if name.lower() == 'resnet50':
        return models.resnet50(weights=True)
    elif name.lower() == 'vgg19':
        return models.vgg19_bn(weights=True)
    elif name.lower() == 'densenet121':
        return models.densenet121(weights=True)
    elif name.lower() == 'googlenet':
        return models.googlenet(weights=True)
    elif name.lower() == 'mobilenet':
        return models.mobilenet_v3_large(weights=True)
    else:
        raise ValueError(f'Model {name} not found')

# Define your Concatenate class
class Concatenate(nn.Module):
    def __init__(self, model1, model2):
        super(Concatenate, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        return torch.cat((x1, x2), dim=1)

# Define your FusedModel class
class FusedModel(nn.Module):
    def __init__(self, concatenated, output):
        super(FusedModel, self).__init__()
        self.concatenated = concatenated
        self.output = output

    def forward(self, x):
        x = self.concatenated(x)
        x = self.output(x)
        return x

# Define a function to create the fused model
def create_fused_model(concatenated, output):
    return FusedModel(concatenated, output)

# Define the output layer
output = nn.Sequential(
    nn.Linear(2000, 100),  # Adjust the input size here
    nn.ReLU(),
    nn.Linear(100, 3),
    nn.Softmax(dim=1)
)

# Instantiate the models
model1 = get_models('mobilenet')
model2 = get_models('googlenet')
concatenated = Concatenate(model1, model2)
fused_model = create_fused_model(concatenated, output)

# Function to load the pre-trained model
def load_model(model_path,multi = True):
    if multi:
        # Load the saved model state dictionary
        with open('newFusedModels/model.pth', 'rb') as f:
            model = torch.load(f,map_location=torch.device('cpu'),pickle_module=dill)
        return model
    else:
        model = torch.load(model_path,map_location=torch.device('cpu'))
        return model



    
def preprocess_image(image_path):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to 224 x 224
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the images
])

    image = read_image(image_path).float()  # Load and convert the image to a float tensor
    image = image.repeat(3, 1, 1)  # Convert the image to 3 channels
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension

    return image
print(torch.cuda.is_available())
device = torch.device('cpu')

# Function to get the output class
def get_output_class(model, image):
    with torch.no_grad():
        output = model(image.to(device))  # Move the image to the appropriate device
        _, predicted = torch.max(output.data, 1)
        predicted_class = predicted.item()

    # Map the predicted class index to the actual class name
    classes = ['Covid19', 'Pneumonia', 'Negative']  # Replace with your actual class names
    output_class = classes[predicted_class]

    return output_class

# Load the pre-trained model
model_path = 'models/FusedModels/GoogLeNet_DenseNet.pth'
model = load_model(model_path,multi=True)
modelName = 'vgg19_model'
model.eval()
imagepath = '5d5d5 (4).jpg'
image = preprocess_image(imagepath)
output_class = get_output_class(model, image)
print(f'The predicted class is: {output_class}')

