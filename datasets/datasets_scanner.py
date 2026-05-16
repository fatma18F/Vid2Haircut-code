import pickle
from torch.nn import functional as F
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import cv2
import os
import random
from PIL import Image
import torch.nn.functional as F
from skimage.transform import resize
import random
from src.utils.preprocessing import erode_mask, normalized_depth_quantile, PILtoTorch
from datasets.real_imgs_overfit_mw import HairstyleRealDataset
from pathlib import Path
import math
import torchvision.transforms as transforms

class HairstyleRealDatasetScanner(HairstyleRealDataset):
    def __init__(self,
                 device='cpu',
                 path_to_meshgrid_data='',
                 image_size=512,
                 gs_scale_path="",
                 gs_views=1,
                 infer_path='',
                 convert_gabor_to_strand_map=False,
                 scene='',
                 sample_from_different_view=False, 
                 use_align=True,
                 use_SingleView=False,
                 use_orientations2=True,
                 use_improvedHairMask=True,
                 use_deformation_mlp=False,
                use_unreal_sim=False
                ):

        super().__init__(device,
                 path_to_meshgrid_data,
                 image_size,
                 gs_scale_path,
                 gs_views,
                 infer_path,
                 convert_gabor_to_strand_map,
                 scene,
                 sample_from_different_view, 
                 use_align)
        self.use_orientations2=use_orientations2
        self.use_improvedHairMask=use_improvedHairMask
        print('HairstyleRealDatasetScanner')
       
    def _setup_image(self, idx, flip=False):
        """L'oad and process orientation map image."""
        img_path = os.path.join(self.root_path, 'orientation_maps512_aligned_aligned', self.hair_list[idx])
        image = np.array(Image.open(img_path))[..., None]

        if flip:
            image = image[:, ::-1]

        image_tensor = torch.tensor(image[:, :, -1:] / 255., device=self.device).float()
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 1, H, W]

        return F.interpolate(image_tensor, size=self.image_size, mode="nearest")[0]
     
        
        
    def use_depth(self, idx, flip=False):
        """Load and normalize depth image."""
        depth_filename = self.hair_list[idx].replace('.png', '.npz').replace('.jpg', '.npz')
        depth_path = os.path.join(self.root_path, 'depth_apple_pro512_aligned_aligned', depth_filename)
        seg_path = os.path.join(self.root_path, 'hair_mask512_aligned_aligned', self.hair_list[idx])

        hair_mask = np.array(Image.open(seg_path)) / 255. > 0.5

        depth_img = normalized_depth_quantile(depth_path, hair_mask=hair_mask, errode=True, kernel=2)
        depth_img = resize(depth_img, (self.image_size, self.image_size), anti_aliasing=True)

        image_tensor = torch.tensor(depth_img[..., None],device=self.device)

        if flip:
            image_tensor = torch.flip(image_tensor, [1])

        return F.interpolate(image_tensor.float()[None].permute(0, 3, 1, 2), size=self.image_size, mode='bilinear')[0]

    
    def use_silh(self, idx, flip=False):
        """Load and blend silhouette images."""
        seg_path = os.path.join(self.root_path, 'hair_mask512_aligned_aligned', self.hair_list[idx])
        body_path = os.path.join(self.root_path, 'body_mask512_aligned_aligned', self.hair_list[idx])

        silh_img = np.array(Image.open(seg_path)) / 255. > 0.5
        silh_img = silh_img[..., None] if silh_img.ndim == 2 else silh_img[:, :, :1]

        image = torch.tensor(silh_img, device=self.device).float()
        hair_silh_np = image.cpu().numpy()

        try:
            body_img = np.array(Image.open(body_path))[:, :, :1] / 255. > 0.5
        except:
            body_img = np.array(Image.open(body_path))[..., None] / 255. > 0.5

        body_tensor = torch.tensor(body_img, device=self.device).float()
        only_body = body_tensor - image
        image += 0.5 * only_body

        full_mask_np = image.cpu().numpy()

        if flip:
            image = torch.flip(image, [1])
            hair_silh_np = hair_silh_np[:, ::-1]
            full_mask_np = full_mask_np[:, ::-1]

        hair_silh_np = (hair_silh_np.squeeze(-1) * 255).astype(np.uint8)
        transformer_mask = full_mask_np.squeeze(-1).copy()

        if image.ndim < 3:
            image = image[:, :, None]

        image_tensor = F.interpolate(image[None].permute(0, 3, 1, 2), size=self.image_size, mode='nearest')[0]
        return image_tensor, transformer_mask
        

    def get_thresholds(self,sample_path):
        # default values
        hair_threshold = None
        bd_threshold = None
        idx=str(sample_path.split('_')[-1].split('.')[0][-2:])


        # idx-based hair_threshold rules
        idx_hair_map = {
            ("20", "25", "75", "80"): 0.25,
            ("30", "40"): 0.30,
            ("35",): 0.35,
            ("45", "00"): 0.20,
            ("50", "55", "60", "65", "70"): 0.15,
        }


        for idx_group, threshold in idx_hair_map.items():
            if idx in idx_group:
                hair_threshold = threshold
                break

  
        bd_threshold=0.15
        return hair_threshold, bd_threshold

    def load_gaus(self, sample_path, flip=False):

        # Paths
        #orientation_mask_sam=angles
        #orient_conf=angles 
        seg_path = os.path.join(self.root_path, 'hair_mask', sample_path)
        silh_path = os.path.join(self.root_path, 'body_mask', sample_path)
        
        conf_orient_path = os.path.join(self.root_path, 'confidence', sample_path)
         
        orient_path = os.path.join(self.root_path, 'orientation_maps', sample_path.replace('.jpg','.png'))
        #orient_path = os.path.join(self.root_path, 'best_ori2', sample_path.replace('.jpg','.png'))
        if not Path(orient_path).exists(): 
             orient_path = os.path.join(self.root_path, 'orientation_maps', sample_path)

        img_path = os.path.join(self.root_path, 'images', sample_path)
        strand_map_path = os.path.join(self.root_path, 'wrpped_strand_map', sample_path)
        depth_path = os.path.join(self.root_path, 'depth_apple_pro', sample_path.replace('.png', '.npz').replace('.jpg', '.npz'))
 
        all_list = sorted(os.listdir(os.path.join(self.root_path, 'hair_mask'))) # originally: hair_mask ???? is it same 
        index = all_list.index(sample_path)
        # print('sample path to change camera', sample_path, index)
        cam = np.load(os.path.join(self.root_path, 'cameras.npy'))[index]
        # Load tensors
        image = PILtoTorch(Image.open(img_path), resolution=None)
        
        mask_body = PILtoTorch(Image.open(silh_path), resolution=None)>0.5
        mask_hair = PILtoTorch(Image.open(seg_path), resolution=None)>0.5
        if self.use_improvedHairMask:
            seg_path=os.path.join(self.root_path, 'improved_hair_mask', sample_path)
            silh_path=os.path.join(self.root_path, 'improved_body_mask', sample_path)
            
            c1 = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE) / 255.
            hair_threshold,bd_threshold =0.5 , 0.5

            mask_hair = torch.from_numpy(np.array(c1))#> hair_threshold
            mask_hair=mask_hair.unsqueeze(dim=-1).permute(2, 0, 1)
            
            c2 = cv2.imread(silh_path, cv2.IMREAD_GRAYSCALE) / 255.
            mask_body = torch.from_numpy(np.array(c2))#> bd_threshold  
            mask_body=mask_body.unsqueeze(dim=-1).permute(2, 0, 1)

            #cv2.imwrite('img1.jpg',  (c1>hair_threshold).astype(np.uint8)*255)   
            #cv2.imwrite('img2.jpg',  (c2>bd_threshold).astype(np.uint8)*255)   


        num_filters=180
        #orient_angle = PILtoTorch(Image.open(orient_path), resolution=None, max_value=num_filters)
        o = torch.from_numpy(np.array(cv2.imread(orient_path, cv2.IMREAD_GRAYSCALE))) 
        o = (180-o)/180  
        orient_angle=o.unsqueeze(dim=-1).permute(2, 0, 1)
        orient_conf = PILtoTorch(Image.open(orient_path), resolution=None,  max_value=num_filters)

         
         
        strand_map = Image.open(strand_map_path).convert('RGB')
        img_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        strand_map = img_to_tensor(strand_map).float() 
        #strand_map = PILtoTorch(Image.open(strand_map_path), resolution=None  ,max_value=180)
        depth = torch.tensor(normalized_depth_quantile(depth_path, hair_mask=mask_hair.numpy()[0], errode=True, kernel=2)[..., None]).permute(2, 0, 1)

        # Preprocess features
        gt_image = image[:3]
#         print(strand_map.shape)
#         input()
        gt_strand_map = strand_map[:3]
        if gt_strand_map.shape[0] == 1:
            gt_strand_map = gt_strand_map.repeat(3,1, 1)
        gt_mask_body = mask_body[:1].float()
        gt_mask_hair = mask_hair[:1].float()
        gt_orient_angle = orient_angle[:1]
        gt_orient_conf = orient_conf[:1]
        gt_depth = depth[:1]
        

#         combine gabor map with direction map from hairstep
        if self.convert_gabor_to_strand_map:
            strand_map_vis = gt_strand_map.permute(1, 2, 0)
            cos_theta = 2 * strand_map_vis[:, :, 1] - 1
            sin_theta = 2 * strand_map_vis[:, :, 2] - 1
            angles = torch.atan2(sin_theta, cos_theta) * 180 / torch.pi % 360

            hairstep_mask_up = ((angles > 90) & (angles <= 270)).clone()

            gabor_angle = (gt_orient_angle * 180)[0].clone()
            gabor_angle = (180 - gabor_angle) % 180
            gabor_mask_up = gabor_angle > 90

            gabor_angle += 180 * ((gabor_mask_up.float() - hairstep_mask_up.float()).abs() > 0).float()

            g_channel = torch.cos(gabor_angle / 180 * torch.pi) / 2 + 0.5
            b_channel = torch.sin(gabor_angle / 180 * torch.pi) / 2 + 0.5

            gabor_map_color = torch.stack([
                strand_map_vis[..., 0],
                g_channel * gt_mask_hair[0],
                b_channel * gt_mask_hair[0]
            ], dim=-1)

            gt_strand_map = gabor_map_color.permute(2, 0, 1)

        # Stack all features
        feats_gaus = torch.cat([
            gt_image, gt_mask_hair, gt_mask_body,
            gt_orient_angle, gt_orient_conf,
            gt_depth, gt_strand_map
        ], dim=0)

        if flip:
            feats_gaus = torch.flip(feats_gaus, dims=[-1])

        # Load camera
        scale_mat = np.eye(4, dtype=np.float32)
        with open(self.gs_scale_path, 'rb') as f:
            transform = pickle.load(f)

            scale_mat[:3, :3] *= transform['scale']
            scale_mat[:3, 3] = transform['translation']

        cam_gaus = cam @ scale_mat
        
   

        return feats_gaus, cam_gaus
        


 