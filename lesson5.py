# Let's start by the basics again

# Imports
import cv2
import os

# Create a folder to save images if it doesn't exist
save_folder = './images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Access the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open the camera")
    exit()

# Read frame from the webcam
ret, frame = cap.read()

if not ret:
    print("Failed to capture an image")
    exit()

# Save the captured image
image_path = os.path.join(save_folder, 'captured_image.jpg')
cv2.imwrite(image_path, frame)

# Release the webcam resource
cap.release()

# Load the saved image to display
saved_image = cv2.imread(image_path)

# Display the saved image
cv2.imshow('Captured Image', saved_image)

# Wait for a key press and then close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
