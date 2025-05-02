
import streamlit as st
import torch
from transformers import pipeline
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from multimodal_sentimental_analysis import MultimodalSentimentModel


st.title("Multimodal Sentiment Analysis")



# Load the model
model = MultimodalSentimentModel()
model.load_state_dict(torch.load("my_multimodal_sentement_classifier_model.pth", map_location="cpu"))
model.eval()


image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#text input
text_input = st.text_area("Enter text:")

#image input
image_input = st.file_uploader("Upload an image...", type=["jpg", "png"])
image_tensor = image_transform(image_input).unsqueeze(0)  # [1, 3, 224, 224]


# Predict
with torch.no_grad():
    output = model(image_tensor, text_input)
    predicted_class = torch.argmax(torch.softmax(output, dim=1), dim=1).item()

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
print("Sentiment:", label_map[predicted_class])