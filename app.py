import streamlit as st
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor

# Load Hugging Face model and processor
model = AutoModelForImageClassification.from_pretrained("poo1123/DRGrading")
image_processor = AutoImageProcessor.from_pretrained("poo1123/DRGrading")

# Streamlit UI components
st.title("Diabetic Retinopathy Image Classification")
st.write("Upload an image to classify the diabetic retinopathy severity.")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    pixel_values = image_processor(image, return_tensors="pt").pixel_values

    # Make prediction
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    # Display result
    class_names = model.config.id2label
    predicted_class = class_names[predicted_class_idx]
    st.write(f"Prediction: {predicted_class} 
