import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image


# Define a function to load your trained model and weights
def load_model():
    model = torch.hub.load('pytorch/vision:v0.13.0', 'resnet50', pretrained=False)
    # Load your trained model weights here (replace with your path and filename)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.LayerNorm(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
#                 nn.Linear(512, 256),
#                 nn.LayerNorm(256),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(p=0.3),
                nn.Linear(128, 50)
                )
    model.load_state_dict(torch.load("weights (1).h5", map_location=device))
    # Freeze the model weights (optional)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model

# Preprocess the image for ResNet-50
def preprocess_image(image):
    
    # Assuming image is a NumPy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)  # Convert to PIL Image
    # Define a transformation for image preprocessing
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to (224, 224)
    transforms.ToTensor(), 
    transforms.Normalize(# Convert the image to a PyTorch tensor
        mean=[0.485, 0.456, 0.406],  # Normalize the image with mean and standard deviation
        std=[0.229, 0.224, 0.225]
    )
    ])
    
    # Preprocess the image
    preprocessed_image = transform(image)
    return preprocessed_image.unsqueeze(0)  # Add batch dimension

# Predict the class using the model
def predict_class(model, image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(preprocessed_image)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

st.title("Medicinal Leaf Classification App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, channels="BGR")
    


    # Load the model if not already loaded
    if "model" not in st.session_state:
        st.session_state["model"] = load_model()

    if st.button("Predict"):
        # Make predictions
        prediction = predict_class(st.session_state["model"], image)
        
        # Download predicted class labels from a separate file (replace with your actual file)
        class_labels = []
        with open("pred_class\dataset_classes.txt", "r") as f:
            class_labels = f.readlines()
        predicted_label = class_labels[prediction].strip()
        
        st.write(f"Predicted Class: {predicted_label}")

st.session_state = st.session_state
