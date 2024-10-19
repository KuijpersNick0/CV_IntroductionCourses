# test program

print("+--------------------------------------+")
print("+     test program koen 17_10_24       +")
print("+     github sync                      +")
print("+--------------------------------------+")

import cv2
import os
from time import sleep
from datetime import datetime
import matplotlib.pyplot as plt

def picture():
    # Start the Pi camera to capture an image 
    print("[INFO] Taking picture...")
    vs = cv2.VideoCapture(1)  # Initialize the camera
    sleep(3.0)  # Let the camera warm up

    # Capture a single frame (image)
    ret, frame = vs.read()  # Read a single frame from the camera
    vs.release()  # Release the camera after capturing the image
 

    # Save the captured image 
    output_filename = datetime.now().strftime("image_%H.%M.%S_%Y-%m-%d.jpg")
    # set destination directory
    os.chdir('C:\Battery_C\project\CV_IntroductionCourses\images\_bottle')

    cv2.imwrite(output_filename, frame)  # Save the image to disk
    print("[INFO] Image saved as {output_filename}")

    # Show the image  
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
    plt.imshow(frame_rgb)  # Display the image
    plt.axis("off")  # Turn off axis labels
    plt.show()  # Show the image in the notebook

def main():
    loop = 0
    while loop < 5:
        loop += 1
        picture()
        print("Print picture :",{loop})
  
#exit camera
cv2.waitKey(0)
cv2.destroyAllWindows() 

  
if __name__=='__main__':
    main()