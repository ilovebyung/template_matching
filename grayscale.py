import cv2
import os
from calculate_angle import calculate_angle
from calculate_angle import rotate_image
from calculate_angle import template_matching

# Define the directory containing JPG files
image_dir = "/home/byungsoo/Pictures/Webcam/5101341"
image_dir = "/home/byungsoo/Pictures/Webcam/5101420"
image_dir = "/home/byungsoo/Pictures/Webcam/backlight"
image_dir = "/home/byungsoo/Pictures/Webcam/fail"

# Convert to grayscale
# Loop through all JPG files in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        # Read the image
        image = cv2.imread(os.path.join(image_dir, filename))

        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Save the grayscale image
        grayscale_filename = os.path.splitext(filename)[0] + "_grayscale.jpg"
        cv2.imwrite(os.path.join(image_dir, grayscale_filename), grayscale_image)

        print(f"Converted and saved: {grayscale_filename}")

# Extract ROI
# Loop through all JPG files in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        # Read the image
        image = cv2.imread(os.path.join(image_dir, filename))

        # Save the extracted image
        grayscale_filename = os.path.splitext(filename)[0] + "_grayscale.jpg"
        cv2.imwrite(os.path.join(image_dir, grayscale_filename), grayscale_image)

        print(f"Converted and saved: {grayscale_filename}")        


        angle = calculate_angle(image, template)
        rotated_image = rotate_image(image, angle=angle)
        plt.imshow(rotated_image, cmap='gray')

        matched_area = template_matching(rotated_image, template)
        plt.imshow(matched_area, cmap='gray')