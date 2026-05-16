
import cv2

def crop_scale_and_resize(image, bbox, target_size, scale=1.0,interpolation=cv2.INTER_LINEAR):
    """
    Crop an image based on a scaled bounding box, add padding, and resize to target size.

    Args:
        image (numpy array): Input image (H, W, C).
        bbox (tuple): Bounding box coordinates (x_min, y_min, x_max, y_max).
        target_size (int): Target size for the output image (e.g., 512).
        scale (float): Scale factor for the bounding box (default is 1.0).

    Returns:
        numpy array: Processed image of size (target_size, target_size, C).
    """
#     print('here img', image.max(), image.min())
    h, w = image.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate the center of the bounding box
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    
    # Calculate the scaled width and height
    width = (x_max - x_min) * scale
    height = (y_max - y_min) * scale
    
    # Compute new bounding box coordinates
    x_min_scaled = int(max(0, cx - width / 2))
    y_min_scaled = int(max(0, cy - height / 2))
    x_max_scaled = int(min(w, cx + width / 2))
    y_max_scaled = int(min(h, cy + height / 2))
    
    # Crop the ROI
    cropped = image[y_min_scaled:y_max_scaled, x_min_scaled:x_max_scaled]
    
    # Calculate padding to make it square
    cropped_h, cropped_w = cropped.shape[:2]
    max_dim = max(cropped_h, cropped_w)
    pad_top = (max_dim - cropped_h) // 2
    pad_bottom = max_dim - cropped_h - pad_top
    pad_left = (max_dim - cropped_w) // 2
    pad_right = max_dim - cropped_w - pad_left
    
    # Add padding to make it square
    padded = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, pad_left, pad_right,
                                borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # Resize to target size
    resized = cv2.resize(padded, (target_size, target_size), interpolation=interpolation)
#     print('out img', resized.max(), resized.min())
    return resized