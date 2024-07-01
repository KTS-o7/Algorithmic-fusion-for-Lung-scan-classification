import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
import os
import pandas as pd

print(os.getcwd())

st.set_page_config(page_title="Lung Scan Classifier", page_icon="🧊", layout="centered")
st.title("Lung Scan Classifier")
st.subheader("Upload a Lung scan image for the classification COVID-19, Tuberculosis, Normal, or Pneumonia")
device = torch.device('cpu')

def load_model(model_path):
    return torch.load(model_path,map_location=torch.device('cpu'))

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

def get_output_class(model, image):
    with torch.no_grad():
        output = model(image.to(device))  # Move the image to the appropriate device
        _, predicted = torch.max(output.data, 1)
        predicted_class = predicted.item()
        st.write(_, predicted, predicted_class)
    # Map the predicted class index to the actual class name
    classes = ['Covid19', 'Pneumonia', 'Negative']  # Replace with your actual class names
    output_class = classes[predicted_class]

    return output_class
def predict_class(modelname = "googlenet", image_file_name:str = None):
    # Load the pre-trained model
    model_path = f'models/Solo Models/{modelname}_model.pth'
    model = load_model(model_path)
    model.eval()
    image = preprocess_image(f"images/{image_file_name}")
    output_class = get_output_class(model, image)
    return output_class

def save_uploaded_file(uploadedfile):
  with open(os.path.join("./images/",uploadedfile.name),"wb") as f:
     f.write(uploadedfile.getbuffer())
     #st.success(f"Saved file :{uploadedfile.name} in images".format(uploadedfile.name))
  return f"/images/{uploaded_file.name}"

uploaded_file = st.file_uploader("Choose a Lung scan image...", type="jpg")

    

if not uploaded_file:
    st.warning("Please upload an image file.")
    
else:
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
        st.image(uploaded_file, caption='Uploaded Lung scan.', use_column_width=True)
    # Apply Function here
        img_path = save_uploaded_file(uploaded_file)
    #st.image(uploaded_file, caption='Uploaded Lung scan.', use_column_width=True)
    model = st.selectbox("Select the models to use", ["googlenet", "densenet121", "vgg19", "resnet","mobilenet"])
    
    if st.button("Classify"):
        st.write("Classifying...")
        output_class = None
        output_class = predict_class(model, image_file_name=uploaded_file.name)
        os.remove(f"images/{uploaded_file.name}")
        st.write(f"The predicted class is: {output_class}")
        st.success("The image was classified successfully.")
        st.balloons()
