# Challenge time! Make a webcam cam editor !

# Make a program that allows to modify in real time the webcam image by pressing keys on the keyboard
# The program must be able to adjust brightness, apply edge detection, flip the image, make the image gray, and blur.
# As a client I want 5 different effects for my webcam application that is going to make me rich !!


import cv2
import os

# Create a folder to save images if it doesn't exist
save_folder = './images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Initialize webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open the camera")
    exit()

# Function to adjust brightness


# Function to apply edge detection
# Hint look at cv2 Canny

# Function to flip the image
 
# Hint
# Handle keypress events with this
# key = cv2.waitKey(1) & 0xFF
 
   