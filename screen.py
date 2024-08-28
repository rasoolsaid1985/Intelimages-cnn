import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model with corrected path
model = tf.keras.models.load_model(r"D:\R\pycharm\PyCharm Community Edition 2024.1.4\project\intel images\mdl_wt (1).hdf5")

# Define the classes
classes = ['buildings', 'forests', 'glacier', 'mountain', 'sea', 'street']

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Streamlit app
st.title("Find out where you are")
st.write("Upload an image and the model will predict")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Make predictions
    predictions = model.predict(img_array)
    st.write(f"Raw model predictions: {predictions}")

    predicted_class = classes[np.argmax(predictions)]
    print(predicted_class)
    
    # Display the prediction
    st.write(f"You are around a: {predicted_class}")

    # Debugging steps
    # Print model summary
    st.write("Model Summary:")
    model_summary_str = []
    model.summary(print_fn=lambda x: model_summary_str.append(x))
    model_summary_str = "\n".join(model_summary_str)
    st.text(model_summary_str)

    # Evaluate the model on a validation set (if available)
    # Note: Replace `validation_ds` with your actual validation dataset
    # Uncomment the following lines if you have a validation dataset
    # val_loss, val_acc = model.evaluate(validation_ds)
    # st.write(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
    
    # Print raw predictions
    st.write("Predictions: ", predictions)
