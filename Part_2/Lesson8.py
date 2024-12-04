# Lesson 8, back to deep learning!
# Let's use a small classification model!

# Introduction to TensorFlow and Keras, two popular DL libraries.

# MobileNetV2 = small convolutional neurel network (CNN) designed for embedded vision
# => Is pre-trained, can be directly used

# To install tensorflow library
# pip install tensorflow

# The model is not that good...

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import os

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Function to capture an image from the webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open the camera")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        cap.release()
        return None

    cap.release()
    return frame

# Function to preprocess the image for the MobileNetV2 model
def preprocess_image(image):
    # Resize image to the input shape required by MobileNetV2 (224x224)
    image_resized = cv2.resize(image, (224, 224))

    # Convert the image to a NumPy array and expand dimensions to match the model's input shape
    image_array = np.expand_dims(image_resized, axis=0)

    # Preprocess the image (normalizes the pixel values for MobileNetV2)
    image_array = preprocess_input(image_array)
    
    return image_array

# Function to classify the image using MobileNetV2
def classify_image(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Perform the prediction
    predictions = model.predict(preprocessed_image)

    # Decode the top 3 predictions
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    return decoded_predictions

# Main logic to capture and classify the image
def main():
    print("Press 'c' to capture and classify an image, 'q' to quit.")

    while True:
        # Wait for the user to press a key
        key = input("Press 'c' to capture and classify or 'q' to quit: ").lower()

        if key == 'c':
            # Capture an image from the webcam
            image = capture_image()

            if image is not None:
                # Classify the captured image
                predictions = classify_image(image)

                # Display the predictions
                print("Top 3 predictions:")
                for i, (imagenet_id, label, score) in enumerate(predictions):
                    print(f"{i+1}: {label} (Confidence: {score * 100:.2f}%)")

                # Show the captured image in a window
                cv2.imshow("Captured Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        elif key == 'q':
            print("Exiting the program.")
            break

if __name__ == '__main__':
    main()

