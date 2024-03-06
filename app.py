import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

def display_labels(labels):
    st.subheader("Plant Leaf Images predictable: ")
    st.write(f"Number of classes: {len(labels)}")
    for label in labels:
        st.write(f"- {label}")
        
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
def predict_class(model, image, threshold):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    class_labels = []
    with open("pred_class\dataset_classes.txt", "r") as f:
        class_labels = f.readlines()    
    # Make predictions
    probabilities = model(preprocessed_image).softmax(dim=1)  # Assuming model outputs probabilities
    prediction = torch.argmax(probabilities, dim=1).item()  # Get the predicted class index
    predicted_label = class_labels[prediction]

    # Check if probability is below the threshold
    if probabilities[0][prediction] < threshold:  # Accessing the probability for the predicted class
        predicted_label = "undetermined"

    return predicted_label

st.title("Medicinal Leaf Classification App")

class_labels = []
with open("pred_class\dataset_classes.txt", "r") as f:
        class_labels = f.readlines()
    
display_labels(class_labels)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, channels="BGR")
    


    # Load the model if not already loaded
    if "model" not in st.session_state:
        st.session_state["model"] = load_model()

    if st.button("Predict"):
        # Make predictions
        
        threshold = 0.7
        prediction = predict_class(st.session_state["model"], image, threshold)
        
        # Download predicted class labels from a separate file (replace with your actual file)
        class_labels = []
        with open("pred_class\dataset_classes.txt", "r") as f:
            class_labels = f.readlines()

        predicted_label = prediction
        
        st.write(f"Predicted Class: {predicted_label}")

st.session_state = st.session_state
