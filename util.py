import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


def time_it(func):
    # This function shows the execution time of
    # the function object passed
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrapper


def remove_noise(image):

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blurring
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply median blurring
    median = cv2.medianBlur(blurred, 5)

    # Return the denoised image
    return median


def erode_and_close_noise(image, threshold, kernel_size):
    """Erodes and closes noise in the image.

    Args:
    image: The image to be processed.
    threshold: The threshold value to be used for thresholding.
    kernel_size: The size of the kernel to be used for erosion and closing.

    Returns:
    The eroded and closed image.
    """

    # Convert the image to grayscale.
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to the image.
    thresholded_image = cv2.threshold(
        grayscale_image, threshold, 255, cv2.THRESH_BINARY)

    # Create a kernel for erosion.
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Erode the image.
    eroded_image = cv2.erode(thresholded_image, kernel, iterations=1)

    # Create a kernel for closing.
    closing_kernel = np.ones(
        (kernel_size * 2 + 1, kernel_size * 2 + 1), np.uint8)

    # Close the image.
    closed_image = cv2.morphologyEx(
        eroded_image, cv2.MORPH_CLOSE, closing_kernel)

    return closed_image


@time_it
def extract_matched_area(image, template):
    """Extract matched area

    Args:
      image: The image to be cropped.
      template: Reference image. 
    Returns:
      aligned and copped image.
    """

    ######## calculate_angle ##########
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

    ######## rotate_image ##########
    # Get the image size
    height, width = image.shape[:2]

    # Get the center of the image
    center = (width // 2, height // 2)

    # Create a rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, int(rotation_deg), 1.0)

    # Warp the image using the rotation matrix
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    ######## template_matching ##########
    h, w = template.shape[::]
    result = cv2.matchTemplate(rotated_image, template, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = min_loc  # Change to max_loc for all except for TM_SQDIFF
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # Crop the image
    matched_area = rotated_image[top_left[1]
        :bottom_right[1], top_left[0]:bottom_right[0]]
    return matched_area

def crop_rectangle(image, threshold_area = 1000000):

    # Apply thresholding to binarize the image
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Detect contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to keep only rectangles with area greater than a threshold
    threshold_area = 1000000  # Set the threshold area

    filtered_contours = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.02, True)
        if len(approx) == 4 and cv2.contourArea(approx) > threshold_area:
            filtered_contours.append(approx)


    # Crop rectangles  
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rectangle = image[y:y+h, x:x+w]
        return rectangle

def crop_area(image):
    cropped_area = image[200:800, 400:1900]
    return cropped_area


if __name__ == "__main__":
    # Load the image.
    file = '0.jpg'
    image = cv2.imread(file)
    image = remove_noise(image)

    '''
    extract matched area
    '''
    file = '2.jpg'
    image = cv2.imread(file, 0)
    file = 'template.jpg'
    template = cv2.imread(file, 0)

    plt.imshow(image, cmap='gray')
    plt.imshow(template, cmap='gray')

    extracted = extract_matched_area(image, template)
    plt.imshow(extracted, cmap='gray')

    cropped = crop_area(extracted)
    plt.imshow(cropped, cmap='gray')
