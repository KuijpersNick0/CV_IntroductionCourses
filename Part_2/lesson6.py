# Lesson 6, start playing with the images

# imports
import cv2
import numpy as np
import os

# Path to the image I want to manipulate
image_path = "./images/captured_image.jpg"

# Load the saved image to manipulate
image = cv2.imread(image_path)

# Function to change brightness
def adjust_brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV for better brightness control
    h, s, v = cv2.split(hsv) 
    v = cv2.add(v, value)  # Add brightness (value can be negative for darker)
    final_hsv = cv2.merge((h, s, v))
    bright_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return bright_img

# Function to crop the image
def crop_image(image, x_start, y_start, width, height):
    return image[y_start:y_start+height, x_start:x_start+width]

# Function to convert the image to grayscale
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to apply binarization (thresholding)
def binarize_image(image, threshold=128):
    gray_image = convert_to_grayscale(image)
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

# Function to apply a convolution filter (e.g., blur)
def apply_convolution_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)

# Manipulate the image using the functions above
bright_image = adjust_brightness(image, 50)  # Increase brightness by 50
cropped_image = crop_image(image, 50, 50, 200, 200)  # Crop a region of the image
gray_image = convert_to_grayscale(image)  # Convert to grayscale
binary_image = binarize_image(image)  # Binarize the image with a threshold
blur_kernel = np.ones((5, 5), np.float32) / 25  # Create a blur kernel
blurred_image = apply_convolution_filter(image, blur_kernel)  # Apply blur filter

# Show the original and manipulated images
cv2.imshow('Original Image', image)
cv2.imshow('Brightened Image', bright_image)
cv2.imshow('Cropped Image', cropped_image)
cv2.imshow('Grayscale Image', gray_image)
cv2.imshow('Binarized Image', binary_image)
cv2.imshow('Blurred Image', blurred_image)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
