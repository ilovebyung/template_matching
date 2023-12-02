import cv2
import matplotlib.pyplot as plt
import numpy as np


def calculate_angle(image, template):
    """
    Calculate angle in reference to template

    Args:
        template: File name of refence image.
        image: File name of input image 

    Returns:
        string: angle of difference between images
    """

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp_template, des_template = sift.detectAndCompute(template, None)
    kp_scene, des_scene = sift.detectAndCompute(image, None)

    # Initialize a BFMatcher with default parameters
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(des_template, des_scene, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract matched keypoints
    src_pts = np.float32(
        [kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calculate homography matrix
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Calculate rotation angle in radians
    rotation_rad = np.arctan2(H[1, 0], H[0, 0])

    # Convert radians to degrees
    rotation_deg = int(np.degrees(rotation_rad))

    print(f"Rotation Angle: {rotation_deg} degrees")
    return str(rotation_deg)


def rotate_image(image, angle):
    """
    Rotate an image by a given angle.

    Args:
        image (numpy.ndarray): The image to be rotated.
        angle (float): The angle of rotation in degrees.

    Returns:
        numpy.ndarray: The rotated image.
    """
    # Get the image size
    height, width = image.shape[:2]

    # Get the center of the image
    center = (width // 2, height // 2)

    # Create a rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, int(angle), 1.0)

    # Warp the image using the rotation matrix
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image


def template_matching(image, template):
    h, w = template.shape[::]
    result = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = min_loc  # Change to max_loc for all except for TM_SQDIFF
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # Crop the image
    matched_area = image[top_left[1]:bottom_right[1],
                         top_left[0]:bottom_right[0]]
    return matched_area
    # cropped_area = matched_area[200:800, 400:1900]
    # return cropped_area


if __name__ == '__main__':
    template = cv2.imread('template.jpg', 0)
    image = cv2.imread('2.jpg', 0)

    plt.imshow(template, cmap='gray')
    plt.imshow(image, cmap='gray')

    angle = calculate_angle(image, template)
    rotated_image = rotate_image(image, angle=angle)
    plt.imshow(rotated_image, cmap='gray')

    matched_area = template_matching(rotated_image, template)
    plt.imshow(matched_area, cmap='gray')
