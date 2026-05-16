import numpy as np

def vis_orient(orient_angle, mask):
    deg = orient_angle * 180  # Convert angles to degrees

    # Calculate color intensities for each orientation
    red = np.clip(1 - np.abs(deg - 0.) / 45., 0, 1) + np.clip(1 - np.abs(deg - 180.) / 45., 0, 1)  # vertical
    green = np.clip(1 - np.abs(deg - 90.) / 45., 0, 1)  # horizontal
    magenta = np.clip(1 - np.abs(deg - 45.) / 45., 0, 1)  # diagonal down
    teal = np.clip(1 - np.abs(deg - 135.) / 45., 0, 1)  # diagonal up

    # Create BGR color components
    red_color = np.array([0, 0, 1])[:, None, None] * red
    green_color = np.array([0, 1, 0])[:, None, None] * green
    magenta_color = np.array([1, 0, 1])[:, None, None] * magenta
    teal_color = np.array([1, 1, 0])[:, None, None] * teal

    # Combine color components into BGR
    bgr = red_color + green_color + magenta_color + teal_color

    # Convert BGR to RGB by reversing channels
    rgb = np.stack([bgr[2], bgr[1], bgr[0]], axis=0)

    # Apply mask
    rgb = rgb * mask

    return rgb