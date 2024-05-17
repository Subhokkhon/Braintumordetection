import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model_path = "inception.h5"  # Assuming inception.h5 is in the same directory as this script
loaded_model = load_model(model_path)

def preprocess_image(image):
    img_resized = image.resize((224, 224))  # Resize the image to match model input shape
    img_rgb = img_resized.convert('RGB')  # Convert image to RGB mode
    img_array = np.array(img_rgb)
    img_normalized = img_array.astype('float32') / 255.0  # Normalize the pixel values
    img_reshaped = np.expand_dims(img_normalized, axis=0)
    return img_reshaped


def predict(image):
    preprocessed_img = preprocess_image(image)
    prediction = loaded_model.predict(preprocessed_img)
    class_names = ["Meningioma", "Glioma", "Pituitary Tumor"]
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

def main():
    st.title('Brain Tumor Detection')

    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Predict the class of the uploaded image
        prediction = predict(image)
        st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()
