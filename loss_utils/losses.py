import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure


def l1(x, y):
    return torch.abs(x-y)


def l2(x, y):
    return ((x-y)**2)



def ssim_orientation_map(angles1, angles2):
    """
    Compute SSIM on orientation maps where each pixel is an angle in degrees (0-180).
    Converts angles to cosine & sine components and computes SSIM separately.
    """
    # Convert degrees to radians
    angles1_rad = torch.deg2rad(angles1*180)
    angles2_rad = torch.deg2rad(angles2*180)

    # Convert to cosine and sine representations
    cos1, sin1 = torch.cos(angles1_rad), torch.sin(angles1_rad)
    cos2, sin2 = torch.cos(angles2_rad), torch.sin(angles2_rad)
#     print('inside', angles1.device, angles2.device, cos1.device)
    
    
    # Initialize SSIM metric
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(angles1.device)

    # Compute SSIM for cos and sin separately

    ssim_cos = ssim_metric(cos1, cos2)
    ssim_sin = ssim_metric(sin1, sin2)

    # Average SSIM scores
    ssim_score = (ssim_cos + ssim_sin) / 2
   
    return ssim_score




def penalty_loss(occupancy_values, epsilon):
    # Compute penalty: max(0, epsilon - occupancy_values)
    penalty = torch.clamp(epsilon - occupancy_values, min=0)
    
    # Average the penalties across all points
    total_penalty = penalty.mean()
    return total_penalty

