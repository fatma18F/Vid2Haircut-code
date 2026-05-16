import cv2
import numpy as np

def postprocess_with_shifted_principal_point(image, c_x, c_y, image_size):
    """
    Postprocess the image to account for the shifted principal point after projection.

    Args:
        image (np.ndarray): The rendered image (2D array).
        c_x (float): x-coordinate of the principal point.
        c_y (float): y-coordinate of the principal point.
        image_size (Tuple[int, int]): Resolution (width, height) of the image.

    Returns:
        np.ndarray: The postprocessed image.
    """
    h, w = image.shape[:2]

    # Shift the image back by subtracting the principal point offset
    # Convert the principal point from image coordinates to normalized device coordinates
    c_x_norm = (c_x - w / 2) / (w / 2)
    c_y_norm = (c_y - h / 2) / (h / 2)

    # Create a translation matrix to shift the image back
    translation_matrix = np.float32([[1, 0, c_x_norm * w], [0, 1, c_y_norm * h]])

    # Apply the translation to the image
    postprocessed_image = cv2.warpAffine(image, translation_matrix, (w, h))

    return postprocessed_image