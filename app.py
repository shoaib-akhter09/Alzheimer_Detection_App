import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Alzheimer Detection",
    layout="wide"
)

st.title("🧠 Alzheimer’s Detection using MobileNetV2")
st.markdown(
    """
    This application uses a **fine-tuned MobileNetV2 deep learning model**
    to classify brain MRI images into Alzheimer’s disease stages.
    """
)

# --------------------------------------------------
# Load trained model (cached for performance)
# --------------------------------------------------
MODEL_PATH = "mobilenetv2_alzheimer_finetuned.h5"

@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_trained_model()
st.success("✅ Model loaded successfully!")

# --------------------------------------------------
# Class labels (must match training order)
# --------------------------------------------------
CLASS_NAMES = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very Mild Dementia"
]

# --------------------------------------------------
# File uploader
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    st.subheader("🩺 Prediction Result")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.subheader("📊 Class Probabilities")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}: {predictions[0][i] * 100:.2f}%")
