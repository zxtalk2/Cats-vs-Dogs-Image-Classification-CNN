import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Website ka Title aur Setup
st.set_page_config(page_title="Cats vs Dogs Classifier", page_icon="🐾")
st.title("🐾 Cats vs Dogs Image Classifier")
st.write("Upload an image of a cat or a dog, and our custom CNN model will predict what it is!")

# 1. Model Load karna (Jo .h5 file aapne upload ki hogi)
@st.cache_resource
def load_my_model():
    # Agar model file ka naam alag hai to yahan change karlein
    return tf.keras.models.load_model('cats_vs_dogs_custom_cnn.h5')

try:
    model = load_my_model()
    st.success("🤖 Model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model. Make sure 'cats_vs_dogs_custom_cnn.h5' is in the same GitHub folder. Error: {e}")

# 2. File Upload karne ka option
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image ko screen par dikhana
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("🧠 Classifying...")

    # 3. Image Preprocessing (Model ke mutabiq 150x150 aur normalize karna)
    img = image.resize((150, 150))
    img_array = np.array(img)
    
    # Check if image is RGB, if grayscale convert to RGB
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
        
    img_array = img_array / 255.0  # Normalization
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension add karna

    # 4. Prediction karna
    prediction = model.predict(img_array)[0][0]

    # 5. Result Show karna
    if prediction > 0.5:
        confidence = prediction * 100
        st.subheader(f"🐕 Prediction: **DOG**")
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        confidence = (1 - prediction) * 100
        st.subheader(f"🐈 Prediction: **CAT**")
        st.write(f"Confidence: {confidence:.2f}%")
