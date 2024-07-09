import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize

# Define your class labels
lesion_classes_dict = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions ',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

# Load your trained model
model = tf.keras.models.load_model('MobilenetV2.h5')

st.title("Skin Disease Classifier")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

if st.button("Classify"):
    img = imread(uploaded_image)
    img = resize(img, (224, 224))  # Resize the image to match the model's input shape
    img = img.astype(np.float32)  # Ensure data type is compatible with the model
    img = np.expand_dims(img, axis=0)

    # Make a prediction using your model
    prediction = model.predict(img)[0]
    predicted_class_index = np.argmax(prediction)# Assuming you get a single prediction for the uploaded image
    predicted_class = lesion_classes_dict[predicted_class_index]
    # Display the predicted class and confidence scores
    st.write("Predicted Class:", lesion_classes_dict[np.argmax(prediction)])

    if(predicted_class == 'Basal cell carcinoma' or predicted_class == 'Melanoma'):
        st.write("It causes Skin Cancer")
    else:
        st.write("It is not cancerous and can be cured.")
    st.write("Confidence Scores:")
    for i, class_name in lesion_classes_dict.items():
        st.write(f"{class_name}: {prediction[i]:.4f}")


