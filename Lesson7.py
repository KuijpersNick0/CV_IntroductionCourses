# Challenge time! Make a webcam cam editor !

import cv2
import os

# Create a folder to save images if it doesn't exist
save_folder = './images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open the camera")
    exit()

# Function to adjust brightness
def adjust_brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    final_hsv = cv2.merge((h, s, v))
    bright_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return bright_img

# Function to apply edge detection
def edge_detection(image):
    return cv2.Canny(image, 100, 200)

# Function to flip the image
def flip_image(image, direction):
    return cv2.flip(image, direction)

# Track the currently applied effect
current_effect = None
brightness_value = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Apply current transformations based on keypresses
    edited_frame = frame.copy()

    # Apply the selected effect
    if current_effect == 'grayscale':
        edited_frame = cv2.cvtColor(edited_frame, cv2.COLOR_BGR2GRAY)
    elif current_effect == 'blur':
        edited_frame = cv2.GaussianBlur(edited_frame, (15, 15), 0)
    elif current_effect == 'brightness':
        edited_frame = adjust_brightness(edited_frame, brightness_value)
    elif current_effect == 'edge':
        edited_frame = edge_detection(edited_frame)
    elif current_effect == 'flip':
        edited_frame = flip_image(edited_frame, 1)  # Horizontal flip

    # Handle keypress events
    key = cv2.waitKey(1) & 0xFF

    # Set the effect based on keypress
    if key == ord('g'):
        current_effect = 'grayscale'
    elif key == ord('b'):
        current_effect = 'blur'
    elif key == ord('+'):
        current_effect = 'brightness'
        brightness_value += 10  # Increase brightness
    elif key == ord('-'):
        current_effect = 'brightness'
        brightness_value -= 10  # Decrease brightness
    elif key == ord('e'):
        current_effect = 'edge'
    elif key == ord('f'):
        current_effect = 'flip'

    # Save the edited image
    if key == ord('s'):
        image_path = os.path.join(save_folder, 'edited_image.jpg')
        cv2.imwrite(image_path, edited_frame)
        print(f"Image saved as {image_path}")

    # Quit the program
    if key == ord('q'):
        break

    # Display the edited image
    cv2.imshow('Webcam Feed - Press keys to apply effects', edited_frame)

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()