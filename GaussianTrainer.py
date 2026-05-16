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
sys.path.append('./submodules/gaussian-splatting-hair/ext/NeuralHaircut')
import torchvision

sys.path.append('./submodules/gaussian-splatting-hair/src')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from utils_3dgs.loss_utils import l1_loss, weighted_bce_soft,  or_loss,or_lossPI, ssim, or_loss_directed, l2_depth_loss, plot_error_hist ,compute_orientation_error_maps,wrap_undirected_diff,undirected_vec,orientation_loss_undirected

from gaussian_renderer import render, render_hair, network_gui
from scene import GaussianModel, GaussianModelCurves
from scene.cameras import CameraMini
from dreifus.matrix import Pose, PoseType, CameraCoordinateConvention, Intrinsics

from utils_3dgs.image_utils import vis_orient, vis_directed_orient

sys.path.append('./submodules/gaussian-splatting-hair/ext/NeuralHaircut')
from NeuS.models.dataset import load_K_Rt_from_P
import open3d as o3d

from hair_loss import generate_thin_hair_weight_mask , generate_baby_hair_weight_mask , generate_babyW , background_fp_loss_soft , generate_baby_hair_weight_mask_orient

def normalize_depth_vis(pred_depth, gt_mask,upper_depth_bound):
    depth = (pred_depth*gt_mask )   # [H,W]
    depth[depth>upper_depth_bound]=1
    
    valid = depth[gt_mask > 0]
    if valid.numel() > 0:
                d_min, d_max = valid.min(), valid.max()
                depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
                return depth_norm
    print('hair mask is all 0 ')
    
def normalize_depth(depth, eps=1e-6):
    min_val = depth.min()
    max_val = depth.max()
    range_val = max_val - min_val
    if range_val < eps:
        return torch.zeros_like(depth)  # or torch.ones_like(depth), or depth itself
    return (depth - min_val) / range_val


def scale_matrix(mat, scale_factor):
    mat[0, 0] /= scale_factor
    mat[1, 1] /= scale_factor
    mat[0, 2] /= scale_factor
    mat[1, 2] /= scale_factor
    return mat


def flip_hairstyle(strands):
    # Flip the x-coordinate (horizontal axis)
    strands_flipped = strands.clone()  # Ensure a copy for gradient preservation
    strands_flipped[:, :, 0] = -strands[:, :, 0]  # Flip x-axis
    # y and z coordinates remain unchanged
    return strands_flipped


def obtain_camera(cam, scaling_factor, resolution):

    intrinsics, pose = load_K_Rt_from_P(None, cam[:3, :4])
    intr=Intrinsics(intrinsics[:3, :3])

    pose = torch.from_numpy(pose).float()
    extrinsics = torch.inverse(pose)
    R = np.transpose(extrinsics[:3,:3].cpu().numpy())  # R is stored transposed due to 'glm' in CUDA code
    T = extrinsics[:3, 3].cpu().numpy()

    FoVx = intr.get_fovx(resolution[0])
    FoVy = intr.get_fovy(resolution[1])
    cx = intr.cx
    cy = intr.cy
    return CameraMini(R=R, T=T, FoVx=FoVx, FoVy=FoVy, width=resolution[0], height=resolution[1], cx=cx, cy=cy)


def obtain_camera_sim(cam, scaling_factor, resolution):
        from scene.cameras_sym import CameraMini
        w2c_world=cam

        intrinsics= np.array([
            [3750, 0, 500],
            [0, 3750, 500],
            [0, 0, 1]
        ], dtype=np.float32)
        intr=Intrinsics(intrinsics)

        FoVx = intr.get_fovx(resolution[0])
        FoVy = intr.get_fovy(resolution[1])
        cx = intr.cx
        cy = intr.cy
        return CameraMini(
                #R=R, 
                #T=T,
                w2c=w2c_world, 
                FoVx=FoVx, 
                FoVy=FoVy, 
                width=resolution[0], 
                height=resolution[1], 
                cx=cx, 
                cy=cy,
        )



def apply_colormap_torch(depth, cmap="inferno"):
    """
    depth: [H,W] tensor in [0,1]
    returns: [3,H,W] tensor in [0,1] RGB
    """
    # torchvision has ready-made colormaps
    import torchvision.transforms.functional as F

    # depth must be 0–255 uint8 for one_hot trick
    depth_u8 = (depth.clamp(0,1)*255).long()  # [H,W]

    # get colormap from matplotlib
    import matplotlib
    cmap = matplotlib.cm.get_cmap(cmap, 256)
    cmap_arr = torch.tensor(cmap.colors, dtype=torch.float32, device='cuda')  # [256,4], RGBA

    depth_rgb = cmap_arr[depth_u8]  # [H,W,4]
    depth_rgb = depth_rgb[...,:3]   # drop alpha
    depth_rgb = depth_rgb.permute(2,0,1)      # [3,H,W]
    return depth_rgb[None]


        


class GaussianTrainerSimple(nn.Module):
    def __init__(self,
                 dataset=None,
                 opt=None,
                 pipe=None,
                 pointcloud_path=None, 
                 ip=None,
                 port=None,
                 scale_matx_path=None,
                 device=None):
        
        
        super().__init__()
        self.scale_matx_path=scale_matx_path
        with open(scale_matx_path, 'rb') as f:
            transform = pickle.load(f)

        self.transform = transform
        self.translate_to_sphere = torch.tensor(transform['translation'], device=device).float()
        self.scale_to_sphere = torch.tensor(transform['scale'], device=device).float()
        
        self.first_iter = 0
        
        self.pipe = pipe
        
        self.opt = opt
        
        self.gaussians = GaussianModel(3, device=device)

        self.gaussians.load_ply_mesh(pointcloud_path, scale_transform=transform)

        self.spatial_lr_scale = 1. if self.gaussians.spatial_lr_scale == 0 else self.gaussians.spatial_lr_scale

        self.device = device
        
        with torch.no_grad():
            # Head gaussians data
            self.gaussians.mask_precomp = self.gaussians.get_label[..., 0] < 0.5
            self.gaussians.xyz_precomp = self.gaussians.get_xyz[self.gaussians.mask_precomp].detach()
            self.gaussians.opacity_precomp = self.gaussians.get_opacity[self.gaussians.mask_precomp].detach()
            self.gaussians.scaling_precomp = self.gaussians.get_scaling[self.gaussians.mask_precomp].detach()
            self.gaussians.rotation_precomp = self.gaussians.get_rotation[self.gaussians.mask_precomp].detach()
            self.gaussians.cov3D_precomp = self.gaussians.get_covariance(1.0)[self.gaussians.mask_precomp].detach()
            self.gaussians.shs_view = self.gaussians.get_features[self.gaussians.mask_precomp].detach().transpose(1, 2).view(-1, 3, (self.gaussians.max_sh_degree + 1)**2)

        bg_color = [0, 0, 0, 0, 0, 0, 0, 0, 0, 7]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=device)
        
        # Start GUI server, configure and run training
        network_gui.init(ip, port)
        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, self.gaussians, self.pipe, self.background, self.scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, self.dataset.source_path)
            except Exception as e:
                network_gui.conn = None

                
    def render(self, cam_sample, scaling_factor, sample_flip, resolution, gt_mask):

        imgs_all = []
        masks_all = []
        depths_all = []
        
        
        n_views= cam_sample.shape[0]
        
        for view in range(n_views):
            camera_gs = obtain_camera(cam_sample[view].detach().cpu().numpy(), scaling_factor, resolution=resolution)
            
            try:
                render_pkg = render(camera_gs, self.gaussians,  self.pipe, self.background, render_direction=self.opt.render_direction)
                
            except Exception as e:
                
                render_pkg = render(camera_gs, self.gaussians,  self.pipe, self.background, render_direction=True)
                
            image = render_pkg["render"]
            mask = render_pkg["mask"]      
            depth_pred = render_pkg['depth']

        
            if sample_flip.item() > 0:
                image = torch.flip(image, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
                depth_pred = torch.flip(depth_pred, dims=[-1])
                
            imgs_all.append(image)
            masks_all.append(mask)
            depths_all.append(depth_pred)

        return torch.stack(imgs_all), torch.stack(masks_all), torch.stack(depths_all)
        
  
    def step(self, gt_cam, gt_image, gt_mask, gt_depth, scaling_factor, idx=0, iteration=0, tb_writer=None, mode='train', flip=None):
        # gt_mask = 2, image = 3, depth =1 
        bs = gt_image.shape[0]
        nviews=gt_cam.shape[1]
        
        images_pred = []
        mask_pred = []
        depths_pred = []


        for idx in range(bs):
            
            resolution = gt_image.shape[-2:]

            resolution = (resolution[1], resolution[0]) 

            cam_sample = gt_cam[idx]
            sample_flip = flip[idx]

            image, mask, depth = self.render(cam_sample,  scaling_factor, sample_flip, resolution, gt_mask[idx])
            
            images_pred.append(image.unsqueeze(0))
            mask_pred.append(mask.unsqueeze(0))
            depths_pred.append(depth.unsqueeze(0))

        
        images_pred = torch.cat(images_pred, dim=0).reshape(bs*nviews, gt_image.shape[-3], gt_image.shape[-2], gt_image.shape[-1]) 
        mask_pred= torch.cat(mask_pred, dim=0).reshape(bs*nviews, gt_mask.shape[-3], gt_mask.shape[-2], gt_mask.shape[-1]) 
        depths_pred  = torch.cat(depths_pred, dim=0).reshape(bs*nviews, gt_depth.shape[-3], gt_depth.shape[-2], gt_depth.shape[-1]) 

        
        gt_image = gt_image.reshape(bs*nviews, gt_image.shape[-3], gt_image.shape[-2], gt_image.shape[-1]) 
        gt_mask = gt_mask.reshape(bs*nviews, gt_mask.shape[-3], gt_mask.shape[-2], gt_mask.shape[-1]) 
        gt_depth = gt_depth.reshape(bs*nviews, gt_depth.shape[-3], gt_depth.shape[-2], gt_depth.shape[-1]) 

        Ll1 = l1_loss(images_pred, gt_image)
        Lssim = (1.0 - ssim(images_pred, gt_image))
        Lmask = l1_loss(mask_pred, gt_mask)
        Ldepth = l1_loss(depths_pred, gt_depth)
        

        if tb_writer is not None:
            
            image = torch.clamp(images_pred, 0.0, 1.0)
            mask = torch.clamp(mask_pred, 0.0, 1.0)

            gt_image = torch.clamp(gt_image, 0.0, 1.0)
            gt_mask = torch.clamp(gt_mask, 0.0, 1.0)
            gt_depth = torch.clamp(gt_depth, 0.0, 1.0)

            tb_writer.add_images(f"{mode}/render", images_pred[0][None], global_step=iteration)
            tb_writer.add_images(f"{mode}/render_mask", F.pad(mask_pred[0], (0, 0, 0, 0, 0, 3-mask_pred[0].shape[0]), 'constant', 0)[None], global_step=iteration)
            tb_writer.add_images(f"{mode}/render_depth", depths_pred[0][None], global_step=iteration)
            tb_writer.add_images(f"{mode}/ground_truth", gt_image[0][None], global_step=iteration)
            tb_writer.add_images(f"{mode}/ground_truth_mask", F.pad(gt_mask[0], (0, 0, 0, 0, 0, 3-gt_mask[0].shape[0]), 'constant', 0)[None], global_step=iteration)
            tb_writer.add_images(f"{mode}/ground_truth_depth", gt_depth[0][None], global_step=iteration)

        return Ll1,  Lssim, Lmask, Ldepth
    

    

class GaussianTrainer(nn.Module):
    def __init__(self,
                 dataset=None,
                 opt=None,
                 opt_hair=None,
                 pipe=None,
                 pointcloud_path_head=None, 
                 ip=None,
                 port=None,
                 gaussian_width=0.001,
                 scale_matx_path=None,
                 use_conf=False,
                 #use_conf2=False,
                 use_directed_loss=False, 
                 loss_type='min',
                 optimize_appearance=False,
                 device=None,
                 pointcloud_path=None,
                 optimize_scale=None,
                 hair_silh_only=None,
                 use_babyW= False,
                 fine_opt=False,
                 use_unreal_sim=False,
                 ):
        
        
        super().__init__()
        self.scale_matx_path=scale_matx_path
        with open(scale_matx_path, 'rb') as f:
            transform = pickle.load(f)
            
        print('gaussian width is', gaussian_width)
        self.loss_type = loss_type
        self.use_directed_loss = use_directed_loss
        self.transform = transform
        self.translate_to_sphere = torch.tensor(transform['translation'], device=device).float()
        self.scale_to_sphere = torch.tensor(transform['scale'], device=device).float()
        
        self.use_conf = use_conf
        self.optimize_appearance = optimize_appearance
        self.first_iter = 0
        
        self.pipe = pipe
        self.dataset = dataset
        
        self.opt = opt
        self.opt_hair = opt_hair
        
        self.gaussians = GaussianModel(dataset.sh_degree, device=device)

        self.gaussians_hair = GaussianModelCurves(dataset.sh_degree, scale=gaussian_width, device=device)
        
           
        # print('pointcloud_path',pointcloud_path)#./example_data/smpl_mesh/000000.ply 
        # self.gaussians.load_ply_mesh(pointcloud_path, scale_transform=self.transform) 


        # self.spatial_lr_scale = 1. if self.gaussians.spatial_lr_scale == 0 else self.gaussians.spatial_lr_scale

        # self.device = device
        
        # with torch.no_grad():
        #     # Head gaussians data
        #     self.gaussians.mask_precomp = self.gaussians.get_label[..., 0] < 0.5
        #     self.gaussians.xyz_precomp = self.gaussians.get_xyz[self.gaussians.mask_precomp].detach()
        #     self.gaussians.opacity_precomp = self.gaussians.get_opacity[self.gaussians.mask_precomp].detach()
        #     self.gaussians.scaling_precomp = self.gaussians.get_scaling[self.gaussians.mask_precomp].detach()
        #     self.gaussians.rotation_precomp = self.gaussians.get_rotation[self.gaussians.mask_precomp].detach()
        #     self.gaussians.cov3D_precomp = self.gaussians.get_covariance(1.0)[self.gaussians.mask_precomp].detach()
        #     self.gaussians.shs_view = self.gaussians.get_features[self.gaussians.mask_precomp].detach().transpose(1, 2).view(-1, 3, (self.gaussians.max_sh_degree + 1)**2)

    
        # # Start GUI server, configure and run training
        # network_gui.init(ip, port)
        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render_hair(custom_cam, self.gaussians, self.gaussians_hair, self.pipe, self.background, self.scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, self.dataset.source_path)
        #     except Exception as e:
        #         network_gui.conn = None

                
    def render(self, cam_sample, selected_strands, scaling_factor, sample_flip, resolution, gt_mask, appearance=None):


        selected_strands = (selected_strands - self.translate_to_sphere ) / self.scale_to_sphere 
        
        features_dc = None
        features_rest = None
        

        if self.optimize_appearance:

            selected_app = appearance
            if sample_flip.item() > 0:
                flipped_app = flip_hairstyle(selected_app)
                
                    
            input_app = flipped_app if sample_flip.item() > 0 else selected_app
#             features_dc = input_app
            features_dc, features_rest = torch.split(input_app, [3, 45], dim=-1)
    

        if sample_flip.item() > 0:
            flipped_strands = flip_hairstyle(selected_strands)
        # strands, baldness mask define which ot optimize
        
        
        input_strands = flipped_strands if sample_flip.item() > 0 else selected_strands
        self.gaussians_hair.update_gaussians_hair(input_strands, features_dc, features_rest)
        
        imgs_all = []
        masks_all = []
        orient_angle_all = []
        orient_conf_all = []
        depths_all = []
        
        
        n_views= cam_sample.shape[0]
        
        for view in range(n_views):
            camera_gs = obtain_camera(cam_sample[view].detach().cpu().numpy(), scaling_factor, resolution=resolution)

            render_pkg = render_hair(camera_gs, self.gaussians, self.gaussians_hair, self.pipe, self.background, render_direction=self.opt.render_direction, use_directed_loss=self.use_directed_loss)

            image = render_pkg["render"]
            mask = render_pkg["mask"]
            orient_angle = render_pkg["orient_angle"]
            orient_conf = render_pkg["orient_conf"]            
            depth_pred = render_pkg['depth']

        
            if sample_flip.item() > 0:
                image = torch.flip(image, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
                orient_angle = torch.flip(orient_angle, dims=[-1])
                orient_conf = torch.flip(orient_conf, dims=[-1])
                depth_pred = torch.flip(depth_pred, dims=[-1]) 
          
            imgs_all.append(image)
            masks_all.append(mask)
            orient_angle_all.append(orient_angle)
            orient_conf_all.append(orient_conf)
            depths_all.append(depth_pred)

        return torch.stack(imgs_all), torch.stack(masks_all), torch.stack(orient_angle_all), torch.stack(orient_conf_all), torch.stack(depths_all)
        
  
    def log_to_tensorboard(self, tb_writer, gt, pred, iteration, mode):
        def clamp(x): return torch.clamp(x, 0.0, 1.0)

        tb_writer.add_images(f"{mode}/render", clamp(pred["image"][0:1]), global_step=iteration)
        tb_writer.add_images(f"{mode}/render_mask", F.pad(pred["mask"][0], (0, 0, 0, 0, 0, 3 - pred["mask"].shape[1]), 'constant', 0)[None], global_step=iteration)
        
       
        tb_writer.add_images(f"{mode}/render_depth", clamp(pred["depth"][0:1]), global_step=iteration)
        tb_writer.add_images(f"{mode}/ground_truth", clamp(gt["image"][0:1]), global_step=iteration)
        tb_writer.add_images(f"{mode}/ground_truth_mask", F.pad(gt["mask"][0], (0, 0, 0, 0, 0, 3 - gt["mask"].shape[1]), 'constant', 0)[None], global_step=iteration)
        tb_writer.add_images(f"{mode}/ground_truth_depth", clamp(gt["depth"][0:1]), global_step=iteration)
        
        #log colored depths
        depth_color = apply_colormap_torch(pred["depth"][0,0], cmap="inferno")  # [3,H,W], float32
        tb_writer.add_images(f"{mode}/render_colored_depth", depth_color, global_step=iteration)
        gt_depth_color = apply_colormap_torch(gt["depth"][0,0], cmap="inferno")  # [3,H,W], float32
        tb_writer.add_images(f"{mode}/ground_truth_colored_depth", gt_depth_color, global_step=iteration)

        if self.use_conf:
            conf_vis = (1 - 1 / (pred["orient_conf"][0][0] + 1)) * pred["mask"][0, :1]
            gt_conf_vis = (1 - 1 / (gt["orient_conf"][0][0] + 1)) * gt["mask"][0, :1]
            tb_writer.add_images(f"{mode}/render_conf", vis_orient(pred["orient_angle"][0], conf_vis)[None], global_step=iteration)
            tb_writer.add_images(f"{mode}/ground_truth_conf", vis_orient(gt["orient_angle"][0], gt_conf_vis)[None], global_step=iteration)

        if self.use_directed_loss and gt["directed_map"] is not None:
            tb_writer.add_images(f"{mode}/ground_truth_dir_orient", torch.cat((gt["mask"][0, :1], gt["directed_map"][0, 1:]), 0)[None], global_step=iteration)
            tb_writer.add_images(f"{mode}/pred_dir_orient", pred["mask"][0, :1] * torch.cat((pred["mask"][0, :1], vis_directed_orient(pred["orient_angle"])[0]), 0)[None], global_step=iteration)
            #tb_writer.add_images(f"{mode}/pred_dir_orient", pred["mask"][0, :1] * torch.cat((pred["mask"][0, :1], pred["orient_angle"][0]), 0)[None], global_step=iteration)
        else:
            tb_writer.add_images(f"{mode}/render_orient", vis_orient(pred["orient_angle"][0], pred["mask"][0, :1])[None], global_step=iteration)
            tb_writer.add_images(f"{mode}/ground_truth_orient", vis_orient(gt["orient_angle"][0], gt["mask"][0, :1])[None], global_step=iteration)
        
        
        torchvision.utils.save_image(depth_color, f"{self.savedir}rendering/pred/depth{iteration}.png")
        torchvision.utils.save_image(gt["mask"][0][0], f"{self.savedir}rendering/gt/gt_hair_mask{iteration}.png")
        torchvision.utils.save_image(gt["directed_map"][0], f"{self.savedir}rendering/gt/gt_strand_map{iteration}.png")
        torchvision.utils.save_image( pred["mask"][0, :1] * torch.cat((pred["mask"][0, :1], vis_directed_orient(pred["orient_angle"])[0]), 0), f"{self.savedir}rendering/pred/strand_map_vis{iteration}.png")     
     
    def render_and_parse_feats(self, strands, gt_cam, feats, scaling_factor, flip, appearance=None, savedir=None,iteration=None):
        self.savedir=savedir
        gt_directed_map = None
        if feats.shape[2] == 8:
            gt_image, gt_mask, gt_orient_angle, gt_orient_conf, gt_depth = torch.split(feats, [3, 2, 1, 1, 1], dim=2)
        else:
            gt_image, gt_mask, gt_orient_angle, gt_orient_conf, gt_depth, gt_directed_map = torch.split(feats, [3, 2, 1, 1, 1, 3], dim=2)
        
        gt_mask_proba=gt_mask
        gt_mask=(gt_mask>0.5).float() 

        bs, nviews = gt_image.shape[0], gt_cam.shape[1]
        image_preds, mask_preds, angle_preds, conf_preds, depth_preds,undirected_vectors,rendered_cov2Ds = [], [], [], [], [] , []  , [] 

        for idx in range(bs):
            resolution = gt_image.shape[-2:][::-1]  # (W, H)
            cam_sample = gt_cam[idx]
            strands_sample = strands[idx]
            sample_flip = flip[idx]
  
            image, mask, angle, conf, depth, undirected_vector , rendered_cov2D= self.render(
                cam_sample, strands_sample, scaling_factor, sample_flip,
                resolution, gt_mask[idx], appearance=appearance
            )

            image_preds.append(image.unsqueeze(0))
            mask_preds.append(mask.unsqueeze(0))
            angle_preds.append(angle.unsqueeze(0))
            conf_preds.append(conf.unsqueeze(0))
            depth_preds.append(depth.unsqueeze(0))
            rendered_cov2Ds.append(rendered_cov2D.unsqueeze(0))
            undirected_vectors.append(undirected_vector.unsqueeze(0))

        def reshape(x): return x.reshape(bs * nviews, *x.shape[-3:])
        os.makedirs(os.path.join(savedir, 'rendering'), exist_ok=True)

        normalized_depth = normalize_depth(
            reshape(torch.cat(depth_preds, dim=0)) * (gt_mask[0,:,:1] )
        )

        pred = {
            "image": reshape(torch.cat(image_preds, dim=0)),
            "mask": reshape(torch.cat(mask_preds, dim=0)),
            "orient_angle": reshape(torch.cat(angle_preds, dim=0)),
            "undirected_vector": reshape(torch.cat(undirected_vectors, dim=0)),
            "orient_conf": reshape(torch.cat(conf_preds, dim=0)),
            "depth": normalized_depth ,#reshape(torch.cat(depth_preds, dim=0)),
            "rendered_cov2D": rendered_cov2D[0] ,#reshape(torch.cat(depth_preds, dim=0)),
        }

        gt = {
            "image": reshape(gt_image),
            "mask": reshape(gt_mask),
            "mask_proba": reshape(gt_mask_proba),
            "orient_angle": reshape(gt_orient_angle),
            "orient_conf": reshape(gt_orient_conf),
            "depth": reshape(gt_depth),
            "directed_map": reshape(gt_directed_map) if gt_directed_map is not None else None,
        }
        def clamp(x): return torch.clamp(x, 0.0, 1.0)

        if iteration %50 ==0:
      
            if iteration ==0:
                os.makedirs(os.path.join(savedir, 'rendering/gt'), exist_ok=True)
                os.makedirs(os.path.join(savedir, 'rendering/pred'), exist_ok=True)

                #torchvision.utils.save_image(gt['orient_angle'] , f"{savedir}/rendering/gt/gt_orients{iteration}.png")
                #torchvision.utils.save_image(gt['orient_angle']* gt['mask'][:,0,] , f"{savedir}/rendering/gt/gt_orients_masked{iteration}.png")
            torchvision.utils.save_image(gt["mask"][0][0], f"{self.savedir}rendering/gt/gt_hair_mask{iteration}.png")
            torchvision.utils.save_image(pred['mask'][0, :1], f"{savedir}/rendering/pred/mask{iteration}.png")
            
            torchvision.utils.save_image(pred['image'], f"{savedir}/rendering/pred/render{iteration}.png")
            torchvision.utils.save_image(gt['image'], f"{savedir}/rendering/gt/gt_image{iteration}.png")
            overlay= gt['image'] *0.5 + clamp(pred["image"][0:1]) *0.8
            torchvision.utils.save_image(overlay, f"{savedir}/rendering/pred/overlay{iteration}.png")

            orient_angle = pred["orient_angle"] * (pred['mask'][0, :1])
            orient_angle_vis = vis_orient(pred["orient_angle"][0], pred['mask'][0, :1])
            torchvision.utils.save_image(orient_angle, f"{savedir}/rendering/pred/orients{iteration}.png")
            torchvision.utils.save_image(orient_angle_vis, f"{savedir}/rendering/pred/orients_vis{iteration}.png")
        
        return {"gt": gt, "pred": pred}
        
        
    def compute_losses(self, gt, pred, gt_orient2=None):
        L1 = l1_loss(pred["image"], gt["image"])
        SSIM = 1.0 - ssim(pred["image"], gt["image"])
        Lmask = l1_loss(pred["mask"], gt["mask"])
        Ldepth = l1_loss(pred["depth"], gt["depth"])

        orient_weight = torch.ones_like(gt["mask"][:, :1])
        if self.use_conf:
            orient_weight *= gt["orient_conf"]
        else:
            pred["orient_conf"] = None

        if self.use_directed_loss and gt["directed_map"] is not None:
            Lorient = or_loss_directed(
                vis_directed_orient(pred["orient_angle"]),
                gt["directed_map"][:, 1:],
                pred["orient_conf"],
                weight=orient_weight,
                mask=gt["mask"][:, :1],
                type=self.loss_type
            )
        else:
            Lorient = or_loss(
                pred["orient_angle"],
                gt["orient_angle"],
                pred["orient_conf"],
                weight=orient_weight,
                mask=gt["mask"][:, :1]
            )

        return {"l1": L1, "ssim": SSIM, "mask": Lmask, "orient": Lorient, "depth": Ldepth}
       
            
    def step(self, strands, gt_cam, feats, scaling_factor, idx=0, iteration=0, tb_writer=None, mode='train', flip=None, appearance=None, cam_idxes=None,savedir=None):

        parsed = self.render_and_parse_feats(strands, gt_cam, feats, scaling_factor, flip, appearance, savedir,iteration)
        gt, pred = parsed['gt'], parsed['pred']

        losses = self.compute_losses(gt, pred)

        if tb_writer is not None:
            self.log_to_tensorboard(tb_writer, gt, pred, iteration, mode)

        return losses["l1"], losses["ssim"], losses["mask"], losses["orient"], losses["depth"]

    
    
class GaussianScannerTrainer(GaussianTrainer):
    def __init__(self, 
                 dataset=None,
                 opt=None,
                 opt_hair=None,
                 pipe=None,
                 pointcloud_path_head=None, 
                 ip=None,
                 port=None,
                 gaussian_width=0.008,
                 scale_matx_path=None,
                 use_conf=False,
                 use_directed_loss=False, 
                 loss_type='min',
                 optimize_appearance=False,
                 device=None,
                 pointcloud_path='',
                 optimize_scale=False,
                 hair_silh_only=False,
                 use_babyW= False,
                 fine_opt=False,
                 use_unreal_sim=False,
                 update_only_vis_strands=False
                ):
        
        super().__init__(dataset,
                         opt,
                         opt_hair,
                         pipe,
                         pointcloud_path_head, 
                         ip,
                         port,
                         gaussian_width,
                         scale_matx_path,
                         use_conf,
                         use_directed_loss, 
                         loss_type,
                         optimize_appearance,
                         device,
                         pointcloud_path,
                         optimize_scale,
                         hair_silh_only,
                         use_babyW,
                         fine_opt,
                         use_unreal_sim,
                        )
        

        self.optimize_scale = optimize_scale
        self.hair_silh_only = hair_silh_only
        self.use_babyW= use_babyW
        self.fine_opt=fine_opt
        self.use_unreal_sim=use_unreal_sim

        self.device=device
        self.update_only_vis_strands=update_only_vis_strands
        
        #self.scale_head = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=optimize_scale)
        self.scale_head = torch.nn.Parameter(torch.tensor(0.8, dtype=torch.float32), requires_grad=optimize_scale)

           
        self.gaussians = GaussianModel(3, device=self.device)
        print('pointcloud_path',pointcloud_path) 
        self.gaussians.load_ply_mesh(pointcloud_path, scale_transform=self.transform) 

        with torch.no_grad():
            # Head gaussians data
            self.gaussians.mask_precomp = self.gaussians.get_label2[..., 0] < 0.5
            self.gaussians.xyz_precomp = self.gaussians.get_xyz[self.gaussians.mask_precomp].detach() 
            self.gaussians.opacity_precomp = self.gaussians.get_opacity[self.gaussians.mask_precomp].detach()
            self.gaussians.scaling_precomp = self.gaussians.get_scaling[self.gaussians.mask_precomp].detach()
            self.gaussians.rotation_precomp = self.gaussians.get_rotation[self.gaussians.mask_precomp].detach()
            self.gaussians.cov3D_precomp = self.gaussians.get_covariance(1.0)[self.gaussians.mask_precomp].detach()
        self.gaussians.shs_view = self.gaussians.get_features[self.gaussians.mask_precomp].detach().transpose(1, 2).view(-1, 3, (self.gaussians.max_sh_degree + 1)**2)
        self.baby_hair_weight=None

        bg_color = [
            0.0, 0.0, 0.0,   # RGB background
            0.0, 0.0,        # mask background
            0.0, 0.0, 0.0,   # cov2D background
            0.0,             # orient_conf background
            7.0              # depth background
        ]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=device)
        
        # Start GUI server, configure and run training
        network_gui.init(ip, port)
        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_hair(custom_cam, self.gaussians, self.gaussians_hair, self.pipe, self.background, self.scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, self.dataset.source_path)
            except Exception as e:
                network_gui.conn = None   




    def render_hair(self, camera_gs):

    	return render_hair(
                camera_gs, self.gaussians, self.gaussians_hair,
                self.pipe, self.background,
                render_direction=self.opt.render_direction,
                use_directed_loss=self.use_directed_loss,
        )  
                 
                        
    def render(self, cam_sample, selected_strands, scaling_factor, sample_flip, resolution, gt_mask, appearance=None):

        selected_strands = (selected_strands - self.translate_to_sphere ) / self.scale_to_sphere             
            
        features_dc = None
        features_rest = None
        

        if self.optimize_appearance:

            selected_app = appearance
            if sample_flip.item() > 0:
                flipped_app = flip_hairstyle(selected_app)
                
                    
            input_app = flipped_app if sample_flip.item() > 0 else selected_app
#             features_dc = input_app
            features_dc, features_rest = torch.split(input_app, [3, 45], dim=-1)
    
    
        
        if sample_flip.item() > 0:
            flipped_strands = flip_hairstyle(selected_strands)
        # strands, baldness mask define which ot optimize
        
        
        input_strands = flipped_strands if sample_flip.item() > 0 else selected_strands

        
        if self.optimize_scale:
#             print('scale here', self.scale_head)
            #input_strands *= self.scale_head  
            self.gaussians.xyz_precomp = self.gaussians._xyz_frame_init.detach() * self.scale_head  
        
        self.gaussians_hair.update_gaussians_hair(input_strands, features_dc, features_rest)
        if self.optimize_appearance:
           self.gaussians.shs_view = self.gaussians.get_features[self.gaussians.mask_precomp].transpose(1, 2).view(-1, 3, (self.gaussians.max_sh_degree + 1)**2)

        imgs_all = []
        masks_all = []
        orient_angle_all = []
        orient_conf_all = []
        depths_all = []
        undirected_vector_all = []
        rendered_cov2D_all =[]
        n_views= cam_sample.shape[0]
        
#         print(resolution)
        for view in range(n_views):

            if self.use_unreal_sim:
                camera_gs = obtain_camera_sim(cam_sample[view].detach().cpu().numpy(), scaling_factor, resolution=resolution)
            else:
                camera_gs = obtain_camera(cam_sample[view].detach().cpu().numpy(), scaling_factor, resolution=resolution)

            if self.update_only_vis_strands:
                render_pkg = self.render_hair(camera_gs)
                vis_strands= render_pkg['visible_strands']

                visible_strands_mask =self.vis_strands(input_strands , vis_strands )
                
                input_strands = torch.where(
                    visible_strands_mask[:, None, None],
                    input_strands,
                    input_strands.detach()
                )
                #input_strands_masked= input_strands[visible_strands_mask]
                self.gaussians_hair.update_gaussians_hair(input_strands, features_dc, features_rest)
            

            render_pkg = self.render_hair(camera_gs)
            image = render_pkg["render"]
            mask = render_pkg["mask"]
            orient_angle = render_pkg["orient_angle"]
            orient_conf = render_pkg["orient_conf"]            
            depth_pred = render_pkg['depth']
            undirected_vector = render_pkg['undirected_vector']
            rendered_cov2D= render_pkg['rendered_cov2D']
            vis_strands= render_pkg['visible_strands']
        
            if sample_flip.item() > 0:
                image = torch.flip(image, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
                orient_angle = torch.flip(orient_angle, dims=[-1])
                orient_conf = torch.flip(orient_conf, dims=[-1])
                depth_pred = torch.flip(depth_pred, dims=[-1])
                
           
            imgs_all.append(image)
            masks_all.append(mask)
            orient_angle_all.append(orient_angle)
            orient_conf_all.append(orient_conf)
            depths_all.append(depth_pred)
            undirected_vector_all.append(undirected_vector)
            rendered_cov2D_all.append(rendered_cov2D)
        

        # #for debugging:
        # os.makedirs(os.path.join(self.savedir, 'GS_pointclouds'), exist_ok=True)
        # if self.curr_iter<10:
        #         self.gaussians.save_ply(f'{self.savedir}GS_pointclouds/pcd_{self.curr_iter}.ply', camera_gs, self.curr_iter)
        #         self.gaussians_hair.save_hair_gaussians_to_ply(f'{self.savedir}GS_pointclouds/hair_gaussians_{self.curr_iter}.ply',camera_gs, self.curr_iter , selected_strands)
        #         self.save_vis_strands(input_strands , vis_strands )

        return torch.stack(imgs_all), torch.stack(masks_all), torch.stack(orient_angle_all), torch.stack(orient_conf_all), torch.stack(depths_all), torch.stack(undirected_vector_all), torch.stack(rendered_cov2D_all) 
    

    def compute_losses(self, gt, pred,gt_orient2=None):
            
        # hair_w = (pred["mask"][:, :1]  * gt["mask"][:, :1]).clamp(min=1e-3)   # [1,H,W]
        # face_w = (pred["mask"][:, 1]  * gt["mask"][:, 1]).clamp(min=1e-3)   # [1,H,W]
        # hair_w = hair_w.detach() 
        # face_w = face_w.detach() 
        #L1 = l1_loss(pred["image"], gt["image"], weight=hair_w)
        #L1 += l1_loss(pred["image"], gt["image"], weight=face_w)
        L1=torch.zeros(1, device=pred["image"].device)

        #SSIM = 1.0 - ssim(pred["image"]*hair_w, gt["image"]*hair_w)
        #SSIM += 1.0 - ssim(pred["image"]*face_w, gt["image"]*face_w)
        SSIM=torch.zeros(1, device=pred["image"].device)
        
        ############     Hair mask
        if self.hair_silh_only:
            gt_mask = gt["mask_proba"][:, :1]
            pred_mask = pred["mask"][:, :1]
            self.baby_hair_weight=1
            if self.use_babyW :
                self.babyW ,  self.baby_hair_weight= generate_babyW(self.curr_iter,self.fine_opt, gt["mask"], pred["mask"] )        
                Lmask = l1_loss(pred["mask"][:, :1], gt["mask"][:, :1],self.babyW )
                
            else:
                
                Lmask = l1_loss(pred["mask"][:, :1], gt["mask"][:, :1] )
    
        else:
            Lmask = l1_loss(pred["mask"], gt["mask"])
        

        Ldepth = l2_depth_loss(pred["depth"].squeeze(1), gt["depth"].squeeze(1), mask=gt["mask"][:, :1].squeeze(1))

        orient_weight = torch.ones_like((gt["directed_map"][:, :1]>0).float()  )#gt["mask"][:, :1])
        if self.use_conf:
            orient_weight *= gt["orient_conf"]
        else:
            pred["orient_conf"] = None
        
        Lorient2=0
        if self.use_directed_loss and gt["directed_map"] is not None:

            Lorient = or_loss_directed(
                vis_directed_orient(pred["orient_angle"] ),
                gt["directed_map"][:, 1:],
                pred["orient_conf"],
                weight=orient_weight,
                mask=gt["mask"][:, :1], #(gt["directed_map"][:, :1]>0).float() ,
                type=self.loss_type
            )

            # penalize any non-zero direction outside hair
            G,B= vis_directed_orient(pred["orient_angle"] )[0]  
            gt_mask  =  gt["mask"][:, :1]#>0.5
            pred_mask=pred["mask"][:, :1]#>0.0
            
            with torch.no_grad():
              M_out = (~(gt_mask>0.0)).float() #pred_mask
              if self.curr_iter %50 ==0:
                 torchvision.utils.save_image(gt_mask*(~(pred_mask>0)), f"{self.savedir}rendering/mask_diff{self.curr_iter}.png")
            
        else:
            original_orient_angle=gt['orient_angle'].clamp(0.0, 1.0) 
            orient_angle = pred["orient_angle"].clamp(0.0, 1.0)
            Lorient = or_loss(
                            orient_angle,
                            original_orient_angle,
                            pred["orient_conf"],
                            weight=orient_weight,
                            mask=gt["mask"][:, :1]
                        )


        #gt_orient_angle = #gt_orient2 
        orient_conf=gt["orient_conf"].to(pred["orient_angle"].device)
        gt_orient_angle=gt['orient_angle'].to(pred["orient_angle"].device)

        # Interpolate hair masks and pred orientation map
        mask=gt["mask"][:, :1]
        gt_mask = F.interpolate(mask,
                        size=(gt_orient_angle.shape[2], gt_orient_angle.shape[3]),
                        mode='bilinear',
                        align_corners=False)
        U_pred = F.interpolate(pred["undirected_vector"],
                        size=(gt_orient_angle.shape[2], gt_orient_angle.shape[3]),
                        mode='bilinear',
                        align_corners=False)
        U_pred = F.normalize(U_pred, dim=1, eps=1e-8)

        #GT orientation map 
        gt_orient_rad =gt_orient_angle * math.pi #(180-gt_orient_angle2)/1 * math.pi
        U_gt = torch.cat([torch.cos(2*gt_orient_rad), torch.sin(2*gt_orient_rad)], dim=1)
        U_gt = F.normalize(U_gt, dim=1, eps=1e-8)

        if self.use_conf:
            weight = orient_conf * gt_mask

        else:
            weight = gt_mask

        # Cosine similarity loss
        sim = (U_pred * U_gt).sum(dim=1, keepdim=True).clamp(-1,1)
        loss = 1.0 - sim
        Lorient2 = (loss * weight).sum() / weight.sum().clamp_min(1.0)
        #err_rad = 0.5 * torch.acos(sim)                                        # [B,1,H,W], in [0,π/2]
        #mean_err_deg = (err_rad * (180.0/math.pi) * gt_mask2).sum() / gt_mask2.sum().clamp_min(1.0)

        #for debigging
        # if self.curr_iter %50 ==0:
        #     savedir=self.savedir

        #     gt_image2 = F.interpolate(gt['image'],
        #                                     size=(gt_orient_angle2.shape[2], gt_orient_angle2.shape[3]),
        #                                     mode='bilinear',
        #                                     align_corners=False)                    
        #     hair_m = gt_mask2[:, :1]   # [1,1,H,W]
        #     err_deg, err_heat, overlay = self.heatmap_from_U(U_pred, U_gt, hair_mask=hair_m, rgb=gt_image2[0])

            
        #     os.makedirs(os.path.join(savedir, 'orientations'), exist_ok=True)

        #     cv2.imwrite(f"{savedir}/orientations/orient_err_heat{iteration}.png", cv2.cvtColor(err_heat, cv2.COLOR_RGB2BGR))
        #     if overlay is not None:
        #         cv2.imwrite(f"{savedir}/orientations/orient_err_overlay{iteration}.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        #return {"l1": L1, "ssim": SSIM, "mask": Lmask, "orient": Lorient,"orient2": Lorient2, "depth": Ldepth}
        return {"l1": L1, "ssim": SSIM, "mask": Lmask, "orient": Lorient, "depth": Ldepth}
    
     
    def step(self, strands, gt_cam, feats, scaling_factor, idx=0, iteration=0, tb_writer=None, mode='train', flip=None, appearance=None, cam_idxes=None, savedir=None,gt_orient2=None):
        self.curr_iter=iteration
        self.savedir=savedir
        self.cam_idxes=cam_idxes
        parsed = self.render_and_parse_feats(strands, gt_cam, feats, scaling_factor, flip, appearance,savedir,iteration)
        gt, pred = parsed['gt'], parsed['pred']
    
        
        losses = self.compute_losses(gt, pred,gt_orient2)

        if tb_writer is not None:
            self.log_to_tensorboard(tb_writer, gt, pred, iteration, mode)

        return losses["l1"], losses["ssim"], losses["mask"], losses["orient"], losses["depth"]
    


    
    @torch.no_grad()
    def vis_strands(self, input_strands , hair_visible_front ):
        num_strands, num_points = input_strands.shape[:2]
        segments_per_strand = num_points - 1

        strand_idx = torch.arange(hair_visible_front.numel(), device=hair_visible_front.device) // segments_per_strand
        visible_strands_mask = torch.zeros(num_strands, dtype=torch.bool, device=hair_visible_front.device)
        visible_strands_mask.index_put_((strand_idx,), hair_visible_front, accumulate=True)
        # Pick visible strands
        vis_strands_front = input_strands[visible_strands_mask] 
        return visible_strands_mask


    @torch.no_grad()
    def save_vis_strands(self, input_strands , hair_visible_front ):
        num_strands, num_points = input_strands.shape[:2]
        segments_per_strand = num_points - 1

        strand_idx = torch.arange(hair_visible_front.numel(), device=hair_visible_front.device) // segments_per_strand
        visible_strands_front = torch.zeros(num_strands, dtype=torch.bool, device=hair_visible_front.device)
        visible_strands_front.index_put_((strand_idx,), hair_visible_front, accumulate=True)
        # Pick visible strands
        vis_strands_front = input_strands[visible_strands_front] 

        #torch.save(visible_strands_front,f'{self.savedir}/GS_pointclouds/vis_strands_mask.npy')

        # select only those points
        #vis_points_front = vis_strands_front[point_mask].view(-1, 3)  
        vis_points_front = vis_strands_front.reshape(-1, 3)  
        vis_points=(vis_points_front * self.scale_to_sphere ) + self.translate_to_sphere 
                # [N_pts_front, 3]
        vis_points_front_np = vis_points.detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vis_points_front_np)
        pcd.paint_uniform_color([0.0, 1.0, 0.0])  # blue hair
        o3d.io.write_point_cloud(f'{self.savedir}/GS_pointclouds/vis_strands_cam{self.cam_idxes.item()}.ply', pcd)
        return visible_strands_front







    @torch.no_grad()
    def heatmap_from_U(self,U_pred, U_gt, hair_mask=None, rgb=None, cap_deg=45.0):
        """
        U_pred, U_gt : [1,2,H,W] (cos 2θ, sin 2θ). They can be approx-unit; we re-normalize.
        hair_mask    : [1,1,H,W] in {0,1} (optional). If None, uses ones.
        rgb          : optional image to overlay; [3,H,W] float 0..1 or [H,W,3] uint8.
        cap_deg      : saturate colormap at this error (degrees).
        Returns:
        err_deg [H,W] float32 (masked),
        heat_rgb [H,W,3] uint8,
        overlay [H,W,3] uint8 or None
        """
        # Ensure same spatial size
        H, W = U_pred.shape[-2:]
        if U_gt.shape[-2:] != (H, W):
            U_gt = F.interpolate(U_gt, size=(H, W), mode='bilinear', align_corners=False)

        # Normalize fields (guard against interpolation drift)
        U_pred = F.normalize(U_pred, dim=1, eps=1e-8)
        U_gt   = F.normalize(U_gt,   dim=1, eps=1e-8)

        # Mask
        if hair_mask is None:
            hair_mask = torch.ones(1,1,H,W, device=U_pred.device, dtype=U_pred.dtype)
        elif hair_mask.shape[-2:] != (H, W):
            hair_mask = F.interpolate(hair_mask, size=(H, W), mode='nearest')

        # Cosine similarity in doubled-angle space -> undirected angular error
        sim     = (U_pred * U_gt).sum(dim=1, keepdim=True).clamp(-1, 1)   # [1,1,H,W]
        err_rad = 0.5 * torch.acos(sim)                                    # [1,1,H,W]
        err_deg = (err_rad * (180.0 / math.pi)) * hair_mask                # [1,1,H,W]

        # Heatmap (0..cap_deg -> 0..255) as 2D uint8
        err_norm = torch.clamp(err_deg / cap_deg, 0, 1)[0,0]               # [H,W]
        heat_u8  = (err_norm * 255).byte().cpu().numpy()                   # [H,W]
        heat_rgb = cv2.applyColorMap(heat_u8, cv2.COLORMAP_MAGMA)          # [H,W,3] BGR
        heat_rgb = cv2.cvtColor(heat_rgb, cv2.COLOR_BGR2RGB)

        # Optional overlay
        overlay = None
        if rgb is not None:
            if torch.is_tensor(rgb):
                if rgb.ndim == 3 and rgb.shape[0] == 3:  # [3,H0,W0] float
                    rgb_np = (rgb.clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                else:  # assume [H,W,3]
                    rgb_np = rgb.cpu().numpy()
            else:
                rgb_np = rgb
            if rgb_np.shape[:2] != (H, W):
                rgb_np = cv2.resize(rgb_np, (W, H), interpolation=cv2.INTER_LINEAR)
            overlay = cv2.addWeighted(rgb_np, 0.6, heat_rgb, 0.4, 0)

        return err_deg[0,0].cpu().numpy(), heat_rgb, overlay