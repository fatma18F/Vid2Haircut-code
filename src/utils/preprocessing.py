import numpy as np
import cv2
import torch


def erode_mask(mask, kernel_size=3, iterations=1):
    """
    Erode a binary mask to remove artifacts and smooth edges.
    
    Args:
        mask (np.ndarray): Input binary mask (0 and 1 or 0 and 255).
        kernel_size (int): Size of the erosion kernel.
        iterations (int): Number of times to apply erosion.
        
    Returns:
        np.ndarray: Eroded binary mask.
    """
    # Ensure mask is binary (0 and 255)
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Create a square kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Apply erosion
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=iterations)
    
    # Convert back to binary (0 and 1)
    eroded_mask = (eroded_mask > 0).astype(np.uint8)
    
    return eroded_mask



def normalized_depth_quantile(path, hair_mask, errode=False, kernel=3, lower_quantile=0.02, upper_quantile=0.98):
    """
    Normalize depth values within a hair silhouette, with optional erosion and quantile clipping.
    
    Args:
        depth (np.ndarray): Depth map.
        hair_silh (np.ndarray): Binary silhouette mask (0 and 1 or 0 and 255).
        errode (bool): Whether to erode the silhouette before processing.
        lower_quantile (float): Lower quantile for clipping (e.g., 0.01 for 1%).
        upper_quantile (float): Upper quantile for clipping (e.g., 0.99 for 99%).
        
    Returns:
        np.ndarray: Normalized depth map with optional quantile clipping.
    """
    depth = np.load(path)['depth']
    hair_silh = hair_mask > 0.5
    # Optionally erode the silhouette
    if errode:
        hair_silh = erode_mask(hair_silh, kernel_size=kernel)
        
    # Ensure binary silhouette
    hair_silh = hair_silh > 0
    
    # Extract depth values within the silhouette
    depth_inside_silhouette = depth[hair_silh]
    
    # Compute quantile bounds
    lower_bound = np.quantile(depth_inside_silhouette, lower_quantile)
    upper_bound = np.quantile(depth_inside_silhouette, upper_quantile)
    
    # Clip depth values within the silhouette to the quantile range
    depth_clipped = np.clip(depth_inside_silhouette, lower_bound, upper_bound)
    
    # Normalize depth within the silhouette
    normalized_depth_map = depth.copy()
    normalized_depth_map[hair_silh] = (depth_clipped - lower_bound) / (upper_bound - lower_bound)
    
    # Optionally, set values outside the silhouette to zero or NaN
    normalized_depth_map[~hair_silh] = 0  # or np.nan

    return normalized_depth_map


def PILtoTorch(pil_image, resolution=None, max_value=255.0):

    if resolution is not None:
        resized_image_PIL = pil_image.resize((resolution, resolution))
    else:
        resized_image_PIL = pil_image
        
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / max_value
    
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
