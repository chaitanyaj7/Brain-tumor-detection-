import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model("brain_tumor_cnn_model.h5")

st.title("Brain Tumor Detection 🧠")
st.write("Upload an MRI scan to detect tumor type")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    st.image(uploaded_file, caption='Uploaded MRI', use_container_width=True)


    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"Prediction: **{predicted_class}**")
