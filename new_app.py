import streamlit as st
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img
from PIL import Image

# Load the TFLite model
model = tf.lite.Interpreter('model_vgg16_lc.tflite')
model.allocate_tensors()

# Get input and output tensors
input_details = model.get_input_details()
output_details = model.get_output_details()


def classify_image(image_path, model):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(227, 227))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the input image (according to the VGG16 model)
    img_array /= 255.0
    img_array -= [0.485, 0.456, 0.406]
    img_array /= [0.229, 0.224, 0.225]

    # Set the input tensor
    model.set_tensor(input_details[0]['index'], img_array)

    # Make prediction using the loaded model
    model.invoke()

    # Get the output tensor and convert to class probabilities
    output_data = model.get_tensor(output_details[0]['index'])
    probabilities = tf.nn.softmax(output_data)[0]

    # Return the predicted class label and confidence score
    class_index = tf.argmax(probabilities)
    class_labels =["Ak", "Ala_Idris", "Buzgulu", "Dimnit", "Nazli"]

    class_label = class_labels[class_index]
    confidence_score = probabilities[class_index]

    return class_label, confidence_score


def home():
    st.title("Welcome to Leaf Image Classification !")
    st.write("This app classifies leaves into 5 classes: Ak, Ala_Idris, Buzgulu, Dimnit, and Nazli.")
    st.write("Upload an image of a leaf and the model will predict its type.")
    st.write("Click on the 'Classify' button to initiate the classification process.")


def about():
    st.title("Welcome to our leaf classification model! ")
    st.write("Our model uses the powerful VGG16 architecture to accurately classify leaves into five different categories: Ak, Ala_Idris, Buzgulu, Dimnit, and Nazli. ")
    st.write("With our model, you can quickly and easily identify different types of leaves, making it a valuable tool for botanists, researchers, and nature enthusiasts.")
    st.write("Try it out today and discover the amazing world of leaves!")


def contact():
    st.title("Contact Us")
    st.write("If you have any questions or comments about this website, please feel free to contact us.")
    st.write("Email: contact@leafimageclassification.com")




def main():
    st.set_page_config(page_title="Leaf Image Classification", page_icon="🍃", layout="wide")

    st.sidebar.title("Navigation")
    pages = {"Home": home, "About": about, "Contact": contact}
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Display the selected page
    page = pages[selection]
    page()

    # Add a file uploader widget on the home page
    if selection == "Home":
        uploaded_file = st.file_uploader("Choose an image...",type=["jpg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True, width=300)

            # Add a button to initiate the classification process
            if st.button("Classify"):
                # Call the classify_image function
                class_label, confidence_score = classify_image(uploaded_file, model)

                # Display the predicted class label and confidence score
                st.write(f"Predicted class: {class_label}")
                st.write(f"Confidence score: {confidence_score:.2f}")

if __name__ == '__main__':
    main()
