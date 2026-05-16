import torch
import sys
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torch.nn as nn
import torch
import torchvision.transforms.functional as TF  
import pickle
import math



@torch.no_grad()
def generate_thin_hair_weight_mask(gt_mask_batch, baby_hair_weight=4.0, base_weight=1.0):
    device = gt_mask_batch.device
    weights = torch.full_like(gt_mask_batch, base_weight, dtype=torch.float32)

    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)

    gx = F.conv2d(gt_mask_batch, sobel_x, padding=1)
    gy = F.conv2d(gt_mask_batch, sobel_y, padding=1)
    grad = torch.sqrt(gx.pow(2) + gy.pow(2))

    # normalize gradient
    grad = grad / (grad.max() + 1e-6)

    # soft thin region (no threshold!)
    thin_region = grad * gt_mask_batch

    weights = base_weight + (baby_hair_weight - base_weight) * thin_region

    return weights

def background_fp_loss_soft(pred, gt):
            bg_weight = 1.0 - gt
            return (pred * bg_weight).mean()

@torch.no_grad()
def generate_babyW(curr_iter,fine_opt,gt_mak,pred_mask ) :
  # if cam_idxes ==0:
    if curr_iter <100:  
        baby_hair_weight=1                 
    elif curr_iter <200:  
        baby_hair_weight=5 
    elif curr_iter <400:  
        baby_hair_weight=10 
    elif curr_iter <600: 
        baby_hair_weight=20
    elif curr_iter <900: 
        baby_hair_weight=50
    babyW= generate_baby_hair_weight_mask(fine_opt,gt_mak[:, :1], pred_mask[:, :1], baby_hair_weight=baby_hair_weight, base_weight=1.0)
    return babyW ,  baby_hair_weight


@torch.no_grad()
def generate_baby_hair_weight_mask(fine_opt,gt_mask_batch, pred_mask_tensor ,baby_hair_weight=5.0, base_weight=1.0):
        """
        Generates a weight mask that emphasizes baby hair regions in the ground truth mask.
        gt_mask_batch: (N, 1, H, W) tensor, values are 0 or 1.
        """
        
        device = gt_mask_batch.device
        N, C, H, W = gt_mask_batch.shape

        W_baby_batch = torch.full_like(gt_mask_batch, base_weight, dtype=torch.float32)

        # precompute a vertical mask = 1 on lower half, 0 on upper half
        y = torch.arange(H, device=device).view(H, 1)         # (H,1)
        lower_half_mask = (y >= H // 2).float() 

        for i in range(N):
            gt_mask = gt_mask_batch[i, 0] # (H, W) for current image

            # 1. Edge Detection (Sobel/Canny-like approximation)
            # Use a kernel to detect strong gradients
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
            
            edges_x = F.conv2d(gt_mask.unsqueeze(0).unsqueeze(0), sobel_x, padding=1)
            edges_y = F.conv2d(gt_mask.unsqueeze(0).unsqueeze(0), sobel_y, padding=1)
            
            # Magnitude of gradient
            edges_magnitude = torch.sqrt(edges_x.pow(2) + edges_y.pow(2)).squeeze() # (H, W)
            
            # 2. Thresholding the edges to isolate thin lines
            # These are potential baby hair areas
            baby_hair_candidates = (edges_magnitude > 0.05).float() * gt_mask # Only consider edges that are part of the hair

            # 3. Enhance thin features (Optional: Small dilation/erosion to thin/thicken)
            # A small erosion can help isolate very thin lines
            #kernel = torch.ones(3, 3, device=device).view(1, 1, 3, 3)
            #eroded_mask = F.conv2d(baby_hair_candidates.unsqueeze(0).unsqueeze(0), kernel, padding=1, groups=1)
            #eroded_mask = (eroded_mask > 0).float().squeeze() # Binarize after "erosion"
            
            # Combine with original hair mask to ensure we only target actual hair pixels
            #baby_hair_pixels = eroded_mask * gt_mask * lower_half_mask
            baby_hair_pixels = baby_hair_candidates #* lower_half_mask

            # 4. Assign higher weight to baby hair pixels
            W_baby_batch[i, 0] = torch.where(baby_hair_pixels > 0.5, 
                                            torch.full_like(gt_mask, baby_hair_weight), 
                                            torch.full_like(gt_mask, base_weight))
        
            # 3×3 kernel dilation, iterations=2
            
            if fine_opt:
                w_bg= baby_hair_weight
                w_gap = baby_hair_weight*2
                base_w = torch.ones_like(gt_mask)

                hair_fg=gt_mask.float().unsqueeze(0).unsqueeze(0)
                bg_region    = 1.0 - hair_fg
                false_pos    = pred_mask_tensor * bg_region       # hair where GT says background
                # Regions inside hair where gaps should be preserved:
                # These are pixels where GT = 0 but near GT hair boundary.
                kernel = torch.ones(1,1,3,3, device=hair_fg.device)
                gt_dilated = (F.conv2d(hair_fg, kernel, padding=1) > 0).float()
                gap_region = (gt_dilated - hair_fg).clamp(min=0.0)  # 1-pixel band outside hair
                gap_false  = pred_mask_tensor * gap_region
                W = base_w + (w_bg - 1.0) * (false_pos > 0.05).float()
                W = W + (w_gap - 1.0) * (gap_false > 0.05).float()
                W = W.clamp(max=w_gap)
                W_baby_batch*=W

        return W_baby_batch 


@torch.no_grad()
def generate_baby_hair_weight_mask_orient(gt_mask_batch ,baby_hair_weight=5.0, base_weight=1.0):
        """
        Generates a weight mask that emphasizes baby hair regions in the ground truth mask.
        gt_mask_batch: (N, 1, H, W) tensor, values are 0 or 1.
        """
        
        device = gt_mask_batch.device
        N, C, H, W = gt_mask_batch.shape

        W_baby_batch = torch.full_like(gt_mask_batch, base_weight, dtype=torch.float32)

        # precompute a vertical mask = 1 on lower half, 0 on upper half
        y = torch.arange(H, device=device).view(H, 1)         # (H,1)
        lower_half_mask = (y >= H // 2).float() 

        for i in range(N):
            gt_mask = gt_mask_batch[i, 0] # (H, W) for current image

            # 1. Edge Detection (Sobel/Canny-like approximation)
            # Use a kernel to detect strong gradients
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
            
            edges_x = F.conv2d(gt_mask.unsqueeze(0).unsqueeze(0), sobel_x, padding=1)
            edges_y = F.conv2d(gt_mask.unsqueeze(0).unsqueeze(0), sobel_y, padding=1)
            
            # Magnitude of gradient
            edges_magnitude = torch.sqrt(edges_x.pow(2) + edges_y.pow(2)).squeeze() # (H, W)
            
            # 2. Thresholding the edges to isolate thin lines
            # These are potential baby hair areas
            baby_hair_candidates = (edges_magnitude > 0.1).float() * gt_mask # Only consider edges that are part of the hair

            # 3. Enhance thin features (Optional: Small dilation/erosion to thin/thicken)
            # A small erosion can help isolate very thin lines
            kernel = torch.ones(3, 3, device=device).view(1, 1, 3, 3)
            eroded_mask = F.conv2d(baby_hair_candidates.unsqueeze(0).unsqueeze(0), kernel, padding=1, groups=1)
            eroded_mask = (eroded_mask > 0).float().squeeze() # Binarize after "erosion"
            
            # Combine with original hair mask to ensure we only target actual hair pixels
            #baby_hair_pixels = eroded_mask * gt_mask * lower_half_mask
            baby_hair_pixels = baby_hair_candidates #* lower_half_mask

            # 4. Assign higher weight to baby hair pixels
            W_baby_batch[i, 0] = torch.where(baby_hair_pixels > 0.5, 
                                            torch.full_like(gt_mask, baby_hair_weight), 
                                            torch.full_like(gt_mask, base_weight))
                    
        return W_baby_batch 