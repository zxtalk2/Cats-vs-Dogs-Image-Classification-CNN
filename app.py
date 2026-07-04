import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Model Load Karna
model = tf.keras.models.load_model('cats_vs_dogs_custom_cnn.h5')

# 2. Prediction Function
def predict_image(img):
    # Image preprocessing (150x150 aur normalize)
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Model prediction
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        confidence = prediction * 100
        return f"🐕 Prediction: DOG ({confidence:.2f}% Confidence)"
    else:
        confidence = (1 - prediction) * 100
        return f"🐈 Prediction: CAT ({confidence:.2f}% Confidence)"

# 3. Gradio Interface Setup
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🐾 Cats vs Dogs Image Classifier",
    description="Upload an image of a cat or a dog to test the custom CNN model live!"
)

# Launch App
interface.launch()
