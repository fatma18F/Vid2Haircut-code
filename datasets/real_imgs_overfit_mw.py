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


class HairstyleRealDataset(Dataset):
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
                 use_unreal_sim=False):


        self.device = device
        self.image_size = image_size
        self.gs_views = gs_views
        self.gs_scale_path = gs_scale_path
        self.convert_gabor_to_strand_map = convert_gabor_to_strand_map
        self.sample_from_different_view = sample_from_different_view
        self.use_align = use_align
        self.root_path = infer_path
        self.scale_camera_factor = 1
        self.use_unreal_sim=use_unreal_sim


        # Meshgrid and mask paths
        texture_size = 64
        self.path_to_faces_for_each_origin = os.path.join(path_to_meshgrid_data, f'faces_{texture_size}x{texture_size}.pth')
        self.path_to_coords_for_each_origin = os.path.join(path_to_meshgrid_data, f'coords_for_each_origin_{texture_size}x{texture_size}.pth')
        self.path_to_full_mask = os.path.join(path_to_meshgrid_data, f'scalp_mask_{texture_size}x{texture_size}.png')
        #self.path_to_usc_mean_scalp_mask_07_with_back = os.path.join(path_to_meshgrid_data, f'baldness_mask_with_moreback_{texture_size}x{texture_size}.png')
        self.path_to_usc_mean_scalp_mask_07_with_back = os.path.join(path_to_meshgrid_data, f'scalp_back_mask_less_back_{texture_size}x{texture_size}.png')
        self.path_to_blender_scalp_mask = os.path.join(path_to_meshgrid_data, f'blender_bladness_mask_{texture_size}x{texture_size}.png')

        
        self.path_to_meshgrid = os.path.join(path_to_meshgrid_data, 'usc_mean_scalp_mask_07.png')
        self.use_unreal_sim = use_unreal_sim

        # Load hairstyles
        all_imgs = sorted(os.listdir(os.path.join(self.root_path, 'resized_img_aligned')))
        if not self.sample_from_different_view: 
            self.hair_list = [img for img in all_imgs if img in [scene]]
        else:
            if  self.use_unreal_sim: 
                scene=[ "NewLevelSequence.camera_turnaround.0000.jpeg", \
                        "NewLevelSequence.camera_turnaround.0020.jpeg", \
                        "NewLevelSequence.camera_turnaround.0040.jpeg", \
                        "NewLevelSequence.camera_turnaround.0060.jpeg", \
                        "NewLevelSequence.camera_turnaround.0080.jpeg", \
                        "NewLevelSequence.camera_turnaround.0100.jpeg", \
                        "NewLevelSequence.camera_turnaround.0119.jpeg", \
                ]
            else:
                scene=[ "cam_222200037_000000.jpg", \
                    "cam_222200037_000030.jpg", \
                    "cam_222200037_000050.jpg", \
                    "cam_222200037_000060.jpg", \
                    "cam_222200037_000070.jpg", \
                    "cam_222200037_000080.jpg" ]

            self.hair_list = [img for img in all_imgs if img in scene]


        self.num_frames=len(all_imgs)
        self.n_hairstyles = len(self.hair_list)

        print('Setting up basis...')
        self._setup_meshgrid()
        print('Meshgrid setup complete.')
    

    def _setup_meshgrid(self):
        """Load and prepare meshgrid data."""
        self.faces_for_each_origin = torch.load(self.path_to_faces_for_each_origin).long().to(self.device)
        self.coords_for_each_origin = torch.load(self.path_to_coords_for_each_origin).float().to(self.device)

        full_mask_img = torch.tensor(cv2.imread(self.path_to_full_mask) / 255)[:, :, :1].squeeze(-1).to(self.device)
        meshgrid_mask_img = torch.tensor(cv2.imread(self.path_to_meshgrid) / 255)[:, :, :1].squeeze(-1).to(self.device)
        mean_scalp_mask_07_with_back = torch.tensor(cv2.imread(self.path_to_usc_mean_scalp_mask_07_with_back) / 255)[:, :, :1].squeeze(-1).to(self.device)
        blender_scalp_mask = torch.tensor(cv2.imread(self.path_to_blender_scalp_mask) / 255)[:, :, :1].squeeze(-1).to(self.device)

        self.full_meshgrid_mask = F.interpolate(
            meshgrid_mask_img.unsqueeze(0).unsqueeze(0),
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0) * full_mask_img
        
    def __len__(self):
            return self.n_hairstyles

        
       
    def _setup_image(self, idx, flip=False):
        """Load and process orientation map image."""
        img_path = os.path.join(self.root_path, 'orientation_maps_aligned_aligned', self.hair_list[idx])
        image = np.array(Image.open(img_path))[..., None]

        if flip:
            image = image[:, ::-1]

        image_tensor = torch.tensor(image[:, :, -1:] / 255., device=self.device).float()
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 1, H, W]

        return F.interpolate(image_tensor, size=self.image_size, mode="nearest")[0]
     
        
        
    def use_depth(self, idx, flip=False):
        """Load and normalize depth image."""
        depth_filename = self.hair_list[idx].replace('.png', '.npz').replace('.jpg', '.npz')
        depth_path = os.path.join(self.root_path, 'depth_apple_pro_aligned_aligned', depth_filename)
        seg_path = os.path.join(self.root_path, 'seg_aligned_aligned', self.hair_list[idx])

        hair_mask = np.array(Image.open(seg_path)) / 255. > 0.5

        depth_img = normalized_depth_quantile(depth_path, hair_mask=hair_mask, errode=True, kernel=2)
        depth_img = resize(depth_img, (self.image_size, self.image_size), anti_aliasing=True)

        image_tensor = torch.tensor(depth_img[..., None])

        if flip:
            image_tensor = torch.flip(image_tensor, [1])

        return F.interpolate(image_tensor.float()[None].permute(0, 3, 1, 2), size=self.image_size, mode='bilinear')[0]

    
    def use_silh(self, idx, flip=False):
        """Load and blend silhouette images."""
        seg_path = os.path.join(self.root_path, 'seg_aligned_aligned', self.hair_list[idx])
        body_path = self.train_dl.path.join(self.root_path, 'body_img_aligned_aligned', self.hair_list[idx])

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
        


    def load_gaus(self, sample_path, flip=False):
        wo_align = '_wo_align' if self.use_align else ''
        root = self.root_path + wo_align

        # Paths
        seg_path = os.path.join(root, 'sam_hair_mask', sample_path)
        silh_path = os.path.join(root, 'body_mask', sample_path)
        orient_path = os.path.join(root, 'orientation_maps_aligned_aligned', sample_path)
        img_path = os.path.join(root, 'images', sample_path)
        strand_map_path = os.path.join(root, 'orientation_maps_aligned_aligned', sample_path)
        cam_path = os.path.join(root, 'proj_matx_inv', sample_path.split('.')[0] + '.txt')
        depth_path = os.path.join(root, 'depth_apple_pro_mask', sample_path.replace('.png', '.npz').replace('.jpg', '.npz'))
        print('cam_path',cam_path)
        # Load tensors
        image = PILtoTorch(Image.open(img_path), self.image_size)
        mask_body = PILtoTorch(Image.open(silh_path), self.image_size)
        mask_hair = PILtoTorch(Image.open(seg_path), self.image_size)
        orient_angle = PILtoTorch(Image.open(orient_path), self.image_size)
        orient_conf = PILtoTorch(Image.open(orient_path), self.image_size)
        strand_map = PILtoTorch(Image.open(strand_map_path), self.image_size)
        depth = torch.tensor(
            resize(normalized_depth_quantile(depth_path, hair_mask=mask_hair.numpy()[0], errode=True, kernel=2),
                   (self.image_size, self.image_size), anti_aliasing=True)[..., None]
        ).permute(2, 0, 1)

        # Preprocess features
        gt_image = image[:3]
        gt_strand_map = strand_map[:3]
        gt_mask_body = mask_body[:1]
        gt_mask_hair = mask_hair[:1]
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

        cam_gaus = np.loadtxt(cam_path) @ scale_mat
        return feats_gaus, cam_gaus


        
    def __getitem__(self, idx):
     try:
        flip = False

        baldness_mask = self.full_meshgrid_mask

        image = self._setup_image(idx, flip=flip)

        silh_img, transformer_mask = self.use_silh(idx, flip=flip)
        image = torch.cat((image, silh_img[:1]),  0)

        depth_img = self.use_depth(idx, flip=flip)
        image = torch.cat((image, depth_img[:1]),  0)

        
        gaus_feats = []
        gaus_cam = []
        view_sampls = self.gs_views
        if self.gs_views > 1:
            sample_path = self.hair_list[idx]
            g_feats, g_cam = self.load_gaus(sample_path, flip=flip)
            gaus_cam.append(g_cam)
            gaus_feats.append(g_feats)
            view_sampls = self.gs_views-1
            
        
            names = sorted(os.listdir(os.path.join(self.root_path + 'resized_img')))
            print(self.hair_list, names)
            for i in range(view_sampls):
                idx_of_name=random.randint(0, len(names)-1)
                sample_path=names[idx_of_name]

                g_feats, g_cam = self.load_gaus(sample_path, flip=flip)


                gaus_cam.append(g_cam)
                gaus_feats.append(g_feats)

        
        else:
            if self.sample_from_different_view:
                
                # names = sorted(os.listdir(os.path.join(self.root_path, 'resized_img')))
                # #names = [name for name in names if name not in ['cam_222200037_000050.jpg','cam_222200037_000055.jpg' ,'cam_222200037_000050.jpg']]
                # p_zero = 0.6
                # if random.random() < p_zero:
                #     sample_idx = 0
                # else:
                #     sample_idx = random.randint(1, len(names) - 1)                
                # sample_path=names[sample_idx]
                sample_idx=idx
                print('self.hair_list[idx]',self.hair_list[idx])
                sample_path = self.hair_list[idx]


                #print('sample_path',sample_path)
                if self.use_unreal_sim:
                  idx_of_name = int(sample_path.split('.')[-2])
                else:
                  idx_of_name = int(sample_path.split('.')[0].split('_')[-1])
                
            else:
                #print('self.hair_list[idx]',self.hair_list[idx])
                sample_path = self.hair_list[idx]
                sample_idx,idx_of_name=idx,idx

            if sample_idx>0: #not frontal view
                self.convert_gabor_to_strand_map=False    
            g_feats, g_cam = self.load_gaus(sample_path, flip=flip)
            gaus_cam.append(g_cam)
            gaus_feats.append(g_feats)
            
        gaus_feats = torch.stack(gaus_feats)
        gaus_cam = np.stack(gaus_cam)

        # frontal views
        gaus_cam_frontal=[]
        gaus_feats_frontal=[]
        g_feats_frontal, g_cam_frontal = self.load_gaus(self.hair_list[idx], flip=flip)
        gaus_cam_frontal.append(g_cam_frontal)
        gaus_feats_frontal.append(g_feats_frontal)
            
        gaus_feats_frontal = torch.stack(gaus_feats_frontal)
        gaus_cam_frontal = np.stack(gaus_cam_frontal)


        return (
            image,
            baldness_mask[None],
            gaus_feats,
            gaus_cam,
            torch.tensor([int(flip)]),
            transformer_mask,
            torch.tensor([idx_of_name]),
            gaus_feats_frontal,
            gaus_cam_frontal,
            sample_idx,
        )
     except Exception as e:
        print(f"[DataLoader ERROR] Failed at idx={idx}: {e}")
        return None  # Optional: raise e to crash instead
