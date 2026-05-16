# === Standard Library ===
import os
import sys
import random
import argparse
import pickle
import open3d as o3d
# === Third-Party Libraries ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import cv2
import yaml
from pyhocon import ConfigFactory
from collections import OrderedDict
from pytorch3d.ops import knn_points
import json

# === Project-Specific Modules ===
# Append project paths
sys.path.append('./loss_utils')
sys.path.append('./submodules/gaussian-splatting-hair/ext/NeuralHaircut')
sys.path.append('./submodules/gaussian-splatting-hair/src')
from utils_3dgs.general_utils import safe_state
from src.utils.geometry import decode_pca
from datasets.datasets_scanner import HairstyleRealDatasetScanner

# Losses and model utils
from head_sdf_prior import SDFHeadPrior
from head_prior import SDFQuery
from scalp_renderer import ScalpRenderer
from GaussianTrainer import GaussianScannerTrainer
from src.upsampling.utils import calc_strands_similarity
from src.utils.file_utils import file_backup
from src.utils.save_utils import save_strands
from src.utils.geometry import compute_similarity_transform, can2world_transform , compute_similarity_transform0,umeyama_similarity
from PIL import Image

sys.path.append('./submodules/gaussian-splatting-hair/src')
from utils_3dgs.loss_utils import (
    per_region_length_loss,
    global_wasserstein1,
)

from model_utils.get_projector import create_projector_backbone

# Arguments / config handling
from arguments import ModelParams, PipelineParams, OptimizationParams

# Distributed training utilities
from src.utils import distributed as dist

# === Environment and Torch Settings ===
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import trimesh
import torch.nn as nn
            

class BaseScanTrainer(nn.Module): 
          
    
    def _init_basic_config(self, num_steps_coarse, device, ngpus, accumulate_gradients,
                       upsample_hairstyle, upsample_resolution, optimize_appearance, config,
                       unfreeze_time_for_pca):

        self.num_steps_coarse = num_steps_coarse
        self.device = device
        self.ngpus = ngpus
        print('device:', self.device, '| GPUs:', ngpus, '| accumulate_gradients:', accumulate_gradients)

        self.upsample_hairstyle = upsample_hairstyle
        self.blend_func = lambda x: torch.where(x <= 0.9, 1 - 1.63 * x**5, 0.4 - 0.4 * x)
        #__________________________________________________________________________________________________________________________________________________________________________________________
        roots_origins_up_path= config['gaussians'].get('roots_origins_up', '') 

        data = torch.load(roots_origins_up_path)
        if not isinstance(data, torch.Tensor):
          data = torch.from_numpy(np.array(data))
        self.roots_origins_up = data.float().to(self.device)[None]
        self.path_to_coords_for_each_origin = config['gaussians'].get('path_to_coords_for_each_origin', '') 



        self.optimize_appearance = False #optimize_appearance
        self.config = config
        self.num_points = config['dataset'].get('num_points', 100)
        self.resolution_upsample = upsample_resolution
        
        self.initial_head_path = config['gaussians'].get('pointcloud_path_head', '') #self.initial_head_path: ./example_data/flame/000000.ply  
        pointcloud_path_hair = config['gaussians'].get('pointcloud_path_hair', '') #self.initial_head_path: ./example_data/flame/000000.ply  

        gs_scale_path=config['dataset_real'].get('gs_scale_path', '')
        self.scalp_render = ScalpRenderer(head_path= self.initial_head_path ,
                                              #scalp_path= '/home/ayed/monocular-hair-modeling/inputs/example_data72/scalp_all_data_VHAP_attached.obj',
                                              scalp_path= pointcloud_path_hair,
                                              size=(3208, 2200),
                                              gs_scale_path=gs_scale_path ) #todo fix it with my img sizes

        self.accumulate_gradients = accumulate_gradients
        self.unfreeze_time_for_pca = unfreeze_time_for_pca
        
        self.num_frames = config['head_prior'].get('frames', 1)
        self.frame_step = config['head_prior'].get('frame_step', 100)
        self.all_steps = config['visuals_config'].get('num_epochs', 1)

#       pca map
        self.global_mean_path = config['pca_basis'].get('global_mean_path', '')
        self.mean_shape_path = config['pca_basis'].get('mean_shape_path', '')
        self.blend_shape_path = config['pca_basis'].get('blend_shape_path', '')

        self.use_unreal_sim = config['dataset'].get('use_unreal_sim', False)

        print(f'Optimize method: {self.num_frames} frames, {self.frame_step} steps each.')

        self.num_components = 64
        self.appearance = None

    def _init_config(self, config):
        self.scalp_mask_prediction = None
        self.use_gs_feats = config['dataset'].get('use_gs_map', False)
        self.use_scale = config['dataset'].get('use_scale', False)
        self.use_feats = self.use_gs_feats
        visuals_cfg = config['visuals_config']
        self.logging_freq = visuals_cfg['logging_freq']
        self.save_freq = visuals_cfg['save_freq']
        self.eval_freq = visuals_cfg['eval_freq']
        self.pc_freq = visuals_cfg['pc_freq']
        self.num_epochs = visuals_cfg['num_epochs']
        

        loss_cfg = config['loss_config']
        self.finetune_coarse_model = loss_cfg.get('finetune_coarse_model', False)
        self.penetration_weight = loss_cfg.get('penetration_weight', 0.0)
        self.gaus_l1 = loss_cfg.get('gaus_l1_loss', 0)
        self.gaus_ssim = loss_cfg.get('gaus_ssim_loss', 0)
        self.gaus_mask = loss_cfg.get('gaus_mask_loss', 0)
        self.gaus_orient = loss_cfg.get('gaus_orient_loss', 0)
        self.gaus_depth = loss_cfg.get('gaus_depth_loss', 0)
        self.gaus_bald_mask = loss_cfg.get('gaus_bald_mask', 0)
        self.sdf_penalty = loss_cfg.get('sdf_penalty', 0)
        

        self.region_length_loss = loss_cfg.get('region_length_loss', 0)
        self.scalpPartionningPath = loss_cfg.get('scalpPartionningPath', '')
        self.scalpPartionningPath64x64 = loss_cfg.get('scalpPartionningPath64x64', '')
        self.pca_prior_reg= loss_cfg.get('pca_prior_reg', 0)
        self.pca_reg= loss_cfg.get('pca_reg', 0)
        self.pca_reg_front_back= loss_cfg.get('pca_reg_front_back', 0)
        self.canonic_reg= loss_cfg.get('canonic_reg', 0)
        self.motion_reg= loss_cfg.get('motion_reg', 0)
        self.lambda_frame= loss_cfg.get('lambda_frame', 0)        

        self.smoothnes_length_back= loss_cfg.get('smoothnes_length_back', 0)
        self.smoothnes_dir_back= loss_cfg.get('smoothnes_dir_back', 0)        
        self.smoothnes_length_neighbor= loss_cfg.get('smoothnes_length_neighbor', 0)
        self.smoothnes_dir_neighbor= loss_cfg.get('smoothnes_dir_neighbor', 0)

        self.gravity= loss_cfg.get('gravity', 0)        
        self.temporal_smoothing_loss = loss_cfg.get('temporal_smoothing', 0)
        self.temporal_inertial_loss = loss_cfg.get('temporal_inertial', 0)
        self.scale_output = loss_cfg.get('scale_output', False)
        self.dilate_mask = loss_cfg.get('dilate_mask', False)
        self.transformer_mask_size = loss_cfg.get('transformer_mask_size', 32)
        self.learning_rate = config['optconfig']['lr']
        self.weight_decay = config['optconfig'].get('weight_decay', 0.001)
        self.optimizer_type = config['optconfig'].get('optimizer_type', 'adam')
        print('self.optimizer_type',self.optimizer_type)
        
                ### deformation mlp
        self.use_deformation_mlp = config['dataset_real'].get('use_deformation_mlp', False)
        self.use_stage_deform = loss_cfg.get('use_stage_deform', False)
        self.canonical_only_steps = loss_cfg.get('canonical_only_steps', 300)


        scale_stats_path = loss_cfg.get('scale_stats_path', '')
        try:
            if scale_stats_path:
                with open(scale_stats_path, "rb") as file:
                    scale_stats = pickle.load(file)
                self.scale_stats_mean = torch.tensor(scale_stats['mean'], device=self.device).float()
                self.scale_stats_std = torch.tensor(scale_stats['std'], device=self.device).float()
        except Exception as e:
            print(f"Failed to load scale stats from {scale_stats_path}: {e}")
                    
        self.hairstyle_init = None
        self.hairstyle_prev = None
        self.hairstyle_prev_prev = None
        self.colors_save = None
        self.edited_uvmap = None
        self.hairstyle_dirs = None

    
        
    def _init_dataset_sim(self, config, scene, prefix2, world_size, rank, num_workers):
        data_path = config['dataset'].get('data_path', '')
        print('Dataset path:', data_path, '| config:', config['dataset'], '| scene:', scene)
        self.data_path = data_path
        self.scene = scene
        infer_path=f'{data_path}/{prefix2}/'
            
        from datasets.datasets_scanner_custom_syn import HairstyleRealDatasetScanner
        self.real_set_train = HairstyleRealDatasetScanner(
            **config['dataset_real'],
            infer_path=infer_path,
            scene=scene
        )
        self.real_set_train.prefix=prefix2
        self.frame_name = prefix2
        self.num_workers = num_workers

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.real_set_train, num_replicas=world_size, rank=rank
        )
        self.train_dl = data.DataLoader(
            self.real_set_train,
            config['optconfig']['batch_size'],
            sampler=train_sampler,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )
        

    def _init_dataset(self, config, scene, prefix2, world_size, rank, num_workers):
        data_path = config['dataset'].get('data_path', '')
        print('Dataset path:', data_path, '| config:', config['dataset'], '| scene:', scene)
        self.data_path = data_path
        self.scene = scene

       
        use_SingleView=config['dataset_real']['use_SingleView']
        if use_SingleView:
            infer_path=f'{data_path}/SingleView/{prefix2}/'
        else:
            infer_path=f'{data_path}/{prefix2}/'
        
        self.real_set_train = HairstyleRealDatasetScanner(
            **config['dataset_real'],
            infer_path=infer_path,
            scene=scene
        )
        self.real_set_train.prefix=prefix2
        self.frame_name = prefix2
        self.num_workers = num_workers

        self.train_dl = data.DataLoader(
            self.real_set_train,
            config['optconfig']['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )

        
    def _init_roots_and_blend_shapes(self):
        data= torch.load(self.path_to_coords_for_each_origin)
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(np.array(data)) 
        self.roots_origins = data[None].float().to(self.device)

        global_mean_shape = torch.tensor(np.load(self.global_mean_path), device=self.device).float()
        mean_shape_local = torch.tensor(np.load( self.mean_shape_path), device=self.device).float()
        self.mean_shape =  global_mean_shape + mean_shape_local
        self.blend_shapes = torch.tensor(np.load(self.blend_shape_path), device=self.device).float()
        self.texture_size = 64
        

    def _init_encoders(self, config, device, rank):
        
        #fine
        self.projector_type = config['projector'].get('projector_type', '')
        self.lp_enc = create_projector_backbone(self.projector_type, config)     
        self.lp_enc = self.lp_enc.to(device) #nn.parallel.DistributedDataParallel(self.lp_enc.to(device), device_ids=[rank], find_unused_parameters=True)


        #coarse
        projector_type_elow = config['projector_type_elow'].get('projector_type', '')
        ckpt_path_elow = config['lp_encoder_fine']['ckpt_path_elow']
        lp_enc_elow = self.create_coarse_model(projector_type_elow, config, ckpt_path_elow, device, finetune_coarse_model=self.finetune_coarse_model)
        
        if self.finetune_coarse_model:
            #self.lp_enc_elow = nn.parallel.DistributedDataParallel(lp_enc_elow.to(device), device_ids=[rank], find_unused_parameters=True)
            self.lp_enc_elow = lp_enc_elow.to(device)
        else:
            self.lp_enc_elow = lp_enc_elow.to(device)
        
    def _init_deformation(self, config, device, rank):           
        #per-frame latent codes
        num_frames = (self.real_set_train.num_frames)-1
        C_dim=4 # frame latent size
        self.frame_codes = nn.Embedding(num_frames, C_dim)
        self.frame_codes=self.frame_codes.to(device)
        nn.init.normal_(self.frame_codes.weight, mean=0.0, std=0.05)

        #from mv_deform_strands import DeformMLP
        from mv_deform_points import DeformMLP
        num_texels = self.resolution_upsample * self.resolution_upsample
        I_dim = 8   # texel embedding size
        self.deformerMLP = DeformMLP(num_texels=num_texels,
                         C_dim=C_dim,
                         I_dim=I_dim,
                         num_freqs=2,
                         max_disp_tangent=0.0005,
                         max_disp_normal=0.008,
                         device=device).to(device)
        self.canonic_strands =None


        
    def get_frame_code(self, frame_index):
        return self.frame_codes(frame_index)
            
    def _init_gaussian_trainer(self, dataset, opt, opt_hair, pipe, pointcloud_path_head, ip, port, rank, config):
        self.gaus_trainer = GaussianScannerTrainer(
            dataset=dataset,
            opt=opt,
            opt_hair=opt_hair,
            pipe=pipe,
            pointcloud_path_head=config['gaussians'].get('pointcloud_path_head', ''),
            ip=ip,
            port=port + rank,
            gaussian_width=config['gaussians'].get('gaussian_width', 0.008),
            scale_matx_path=config['dataset_real'].get('gs_scale_path', ''),
            use_conf=config['loss_config'].get('use_conf', False),
            use_directed_loss=config['gaussians'].get('use_directed_loss', False),
            loss_type=config['gaussians'].get('loss_type', "min"),
            optimize_appearance=self.optimize_appearance,
            device=self.device,
            pointcloud_path=config['gaussians'].get('pointcloud_path_head', ''),
            optimize_scale=config['gaussians'].get('optimize_scale', False),
            hair_silh_only=config['gaussians'].get('hair_silh_only', False),
            use_babyW= config['loss_config'].get('use_babyW', ''),
            fine_opt= config['loss_config'].get('fine_opt', False),
            use_unreal_sim=config['dataset'].get('use_unreal_sim', False),
            update_only_vis_strands= config['loss_config'].get('update_only_vis_strands', False),
        )
        self.gaus_trainer.babyW= None

        
    def create_coarse_model(self, projector_type_elow, config, ckpt_path_elow, device, finetune_coarse_model):
        
            elow = create_projector_backbone(projector_type_elow, config)
            checkpoint = torch.load(ckpt_path_elow, map_location=device)
            state_dict = checkpoint['lp_enc']

            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                new_key = k.replace('module.', '')  # Remove `module.` prefix
                new_state_dict[new_key] = v


            elow.load_state_dict(new_state_dict)
            elow.to(device)

            if finetune_coarse_model:
                print('finetune coarse model as well')
                elow.train()
            else:
                elow.eval()

            params_number =  sum(param.numel() for param in elow.parameters())
            print(f'load ckpt {ckpt_path_elow} in coarse model with {params_number}')
            return elow     
    

    def combine_similarity_with_pivot_scale(self,s, R, t, pivot, scale):
        s_final = s * scale
        t_final = scale * t + (1.0 - scale) * pivot  # = scale*(t - pivot) + pivot
        return s_final, R, t_final


    def _init_canonical_to_world(self, config):
        self.original_head_path = './inputs/data/head_prior.obj'
        self.initial_head_path = config['gaussians'].get('pointcloud_path_head', '') #self.initial_head_path: ./example_data/flame/000000.ply  
        
        self.initial_head_path72='/home/ayed/prerocessing_input_monocular_hair_modelling/person72/flame/frame_00000.obj'
        world_head72 = trimesh.load(self.initial_head_path72).vertices[:5023,:]
        self.source = trimesh.load(self.original_head_path).vertices # head prior
        source_f=trimesh.load(self.original_head_path).faces
        pivot = trimesh.load(self.original_head_path).centroid

        s, R, t = umeyama_similarity(self.source, world_head72 ,with_scaling=True)
        delta_y=-0.0045 
        delta_z=-0.01  
        t= t + np.array([0.0,delta_y,delta_z])
        head_prior_aligned = (s * (self.source @ R.T)) +t
        pivot_aligned = (s * (pivot @ R.T)) + t 

        scale = 1.2  
        head_prior_aligned = (head_prior_aligned - pivot_aligned) * scale + pivot_aligned
        s, R, t = self.combine_similarity_with_pivot_scale(s, R, t, pivot_aligned, scale)
        matrix = np.eye(4)
        matrix[:3, :3] = s * R
        matrix[:3, 3] = t

        source = trimesh.load(self.initial_head_path72)
        source.apply_transform(matrix)

        self.world_head = trimesh.load(self.initial_head_path).vertices[:5023,:]
        matrix, transformed_vertices, disparity = trimesh.registration.procrustes(
            head_prior_aligned,
            self.world_head, 
            reflection=False, 
            translation=True, 
            scale=True
        )
        t2 = matrix[:3, 3]
        s2 = np.linalg.norm(matrix[:3, 0])
        R2 = matrix[:3, :3] / s2
        head_prior_aligned2 = (s2 * (head_prior_aligned @ R2.T)) +t2


        # #Debugging
        error = np.linalg.norm(head_prior_aligned2 - self.world_head, axis=1)
        print("Mean alignment error:", np.mean(error))
        print("Max error:", np.max(error))
        
        source_W_o3d = o3d.geometry.TriangleMesh()
        source_W_o3d.vertices = o3d.utility.Vector3dVector(head_prior_aligned2)
        source_W_o3d.triangles = o3d.utility.Vector3iVector(source_f)
        o3d.io.write_triangle_mesh(f"{self.savedir}head_prior_aligned.obj", source_W_o3d)

        s_final = s2 * s
        R_final = R2 @ R
        t_final = s2 * (R2 @ t) + t2
        self.scale_can2world = torch.tensor([s_final], device=self.device).float()
        self.R_can2world = torch.tensor(R_final, device=self.device).float()
        self.t_can2world = torch.tensor(t_final, device=self.device).float()
        
    def _setup_dirs_and_writer(self, savedir):
        os.makedirs(savedir, exist_ok=True)
        for mode in ['train', 'test', 'val']:
            os.makedirs(os.path.join(savedir, f'images_{mode}'), exist_ok=True)
            os.makedirs(os.path.join(savedir, f'pointclouds_{mode}'), exist_ok=True)
        os.makedirs(os.path.join(savedir, 'checkpoints'), exist_ok=True)

        self.savedir = savedir
        self.writer = SummaryWriter(log_dir=os.path.join(savedir, 'logs'))
        self.step = 0
        self.epoch = 0

        
    def _upload_sdf_prior(self, config):
#         load sdf functions
            
        if self.sdf_penalty > 0:
            
            self.sdf_head = SDFQuery(sdf_path=config['head_prior'].get('sdf_path', ""),
                                    bounds_min=config['head_prior'].get('bounds_min', ""),
                                    bounds_max=config['head_prior'].get('bounds_max', ""),
                                     grid_size=self.config['head_prior'].get('grid_size', 64)
                                    )

    
    def _xyz_frame_init(self, head_iter_path):
#         update gaussians  head depends on iteration
        self.gaus_trainer.gaussians.load_ply_mesh(head_iter_path, scale_transform=self.gaus_trainer.transform)


        
        
    def update_iteration(self, frame_idx):
    
        replaced_idx = f"{int(frame_idx):06d}"
#         print('update iteration', replaced_idx)
        
        initial_idx = "000000"


        self.sdf_head = SDFQuery(sdf_path=self.config['head_prior'].get('sdf_path', "").replace(initial_idx, replaced_idx),
                    bounds_min=self.config['head_prior'].get('bounds_min', "").replace(initial_idx, replaced_idx),
                    bounds_max=self.config['head_prior'].get('bounds_max', "").replace(initial_idx, replaced_idx),
                    grid_size=self.config['head_prior'].get('grid_size', 64)
                    )
        

        smplx_iter_path = self.config['gaussians'].get("pointcloud_path", "").replace(initial_idx, replaced_idx)
        head_iter_path = self.config['gaussians'].get("pointcloud_path_head", "").replace(initial_idx, replaced_idx)
        
        print('update min max sdf', smplx_iter_path)
#       update gaussians
        self._xyz_frame_init(smplx_iter_path)

#       update transform from canonical to world
        target = trimesh.load(head_iter_path).vertices[:5023,:]
        s, R, t = compute_similarity_transform(self.source, target)
        self.scale_can2world, self.R_can2world, self.t_can2world = torch.tensor([s], device=self.device).float(), torch.tensor(R, device=self.device).float(), torch.tensor(t, device=self.device).float()
        
        self.hairstyle_prev_prev = None
        
        


    def single_step(self, pred_points_vis, batch, batch_idx, world_size, rank, device, global_rank,mode='train'):
                         
        # Unpack batch
        img, baldness_mask, feats, cam, flip, transformer_mask, cam_idxes, gaus_feats_frontal,  gaus_cam_frontal,sample_idx = batch
        
        # Move inputs to the appropriate device
        device_inputs = [img, baldness_mask, cam, flip, transformer_mask]
        device_inputs = [x.to(self.device).to(rank) for x in device_inputs]
        img, baldness_mask, cam, flip, transformer_mask = device_inputs
        feats = feats.to(self.device).to(rank)            

        # Initialize all losses
        def zero_loss():
            return torch.tensor([0.0], device=img.device)

        losses = {
            'sdf': zero_loss(),
            'gaus_l1': zero_loss(),
            'gaus_ssim': zero_loss(),
            'gaus_mask': zero_loss(),
            'gaus_orient': zero_loss(),
            'gaus_depth': zero_loss(),
            'pca_prior': zero_loss(),
            'pca_reg': zero_loss(),
            'pca_reg_front_back': zero_loss(),
            'canonic_reg': zero_loss(),
            'gravity': zero_loss(),
            'smoothnes_length_neighbor': zero_loss(),
            'smoothnes_dir_neighbor': zero_loss(),
            'smoothnes_length_back': zero_loss(),
            'smoothnes_dir_back': zero_loss(),
            'motion_reg': zero_loss(),
            'frame_reg': zero_loss(),
            'per_region_length': zero_loss(),
        }

        
        # Gaussian feature loss
        if self.use_gs_feats:
            tb_writer = self.writer if self.step % self.pc_freq == 0 and rank == 0 else None
            n_strands = pred_points_vis.shape[1]
            n_pts = pred_points_vis.shape[2]            
            if self.scanner_type=='scanner_3d' :
                world_strands =  pred_points_vis.reshape(1, n_strands, n_pts, 3)
            else:
                world_strands = (self.scale_can2world * (self.R_can2world @ pred_points_vis.reshape(-1, 3).T).T + self.t_can2world).reshape(1, n_strands, n_pts, 3)
                #transform roots to world
                self.transfrom_pcdW(cam_idxes)

       
            # Hair initialization
            #if self.hairstyle_init is None:
            self.hairstyle_init = world_strands[0].reshape(-1, 200, 3).detach()
            self.colors_save = torch.cat((torch.rand(world_strands[0].shape[0], 3).unsqueeze(1).repeat(1, self.num_points, 1), torch.ones(world_strands[0].shape[0], self.num_points, 1)), dim=-1).reshape(-1, 4).cpu()
                
            if self.hairstyle_prev_prev is None:
#                 frame t-2
                self.hairstyle_prev_prev  = self.hairstyle_prev.detach() if self.hairstyle_prev is not None else  world_strands[0].reshape(-1, 200, 3).detach()
#                 frame t-1
                self.hairstyle_prev  = world_strands[0].reshape(-1, 200, 3).detach()
                

            l1, ssim, mask, orient, depth  = self.gaus_trainer.step(
                world_strands, cam, feats,
                scaling_factor=self.real_set_train.scale_camera_factor,
                iteration=self.step, tb_writer=tb_writer,
                mode=mode, flip=flip, appearance=self.appearance, 
                cam_idxes=cam_idxes,
                savedir=self.savedir,
            )


                        
        #Regularization Strategy for Unseen PCA Values
        if self.scanner_type=='scanner_3d' : #or ( self.pca_prior_reg * self.pca_reg==0) :
            self.L2_pca_reg_front_back=torch.zeros(1).cuda()
            self.L2_pca_reg=torch.zeros(1).cuda()
            self.L2_pca_prior_loss=torch.zeros(1).cuda()
            self.L_canonic=torch.zeros(1).cuda()
        
        with torch.no_grad():
                map_cpu = torch.load(self.scalpPartionningPath )
                Partition_ID_Map = torch.as_tensor(map_cpu, device=world_strands.device).detach()
        length_map , interested_idxes_up, up_baldness_mask= self.prepare_needed_data_for_region_scalp(gaus_cam_frontal ,gaus_feats_frontal, baldness_mask ,  pred_points_vis  )
                   
        if self.region_length_loss>0:
            #Generate scalp lengthmap

            #Per region length supervision
            loss_region, stats = self.per_region_length_loss_soft(
                    length_map , Partition_ID_Map, hair_mask=up_baldness_mask,
                    # ORDER: (sides, back, middle)
                    means_cm=(26.0, 17.0, 24.0),   #(24.0, 14.0, 26.0),   
                    stds_cm= (3.0,  1.5,  2.5),    #(3.0,  5.5,  2.5),    
                    lows_cm= (20.0, 16.0,  20.0),  #(20.0, 5.5,  22.0),
                    highs_cm=(28.0, 18.0, 26.0),   #(28.0, 16.0, 29.0),
                    w_gauss=1.0, w_bounds=0.3
                )
            w1 = global_wasserstein1(length_map[up_baldness_mask>0], mu=24., sigma=3., lo=15., hi=26.)
            #w1 penalizes global deviations — too many overly short or long strands
            self.per_region_loss = loss_region + 0.1 * w1
            self.log_statistics(stats , up_baldness_mask , Partition_ID_Map, length_map)

        else:
            self.per_region_loss = torch.zeros(1).cuda()  

        if self.gravity>0:
          L_gravity= self.compute_gravity_loss(pred_points_vis,K_segments=15)#15

        L_dir_neighbor, L_length_neighbor = self.neighbor_smoothness_loss(pred_points_vis,K=20)
        back_L_length = self.back_smoothness_loss(pred_points_vis, Partition_ID_Map, interested_idxes_up)

        #deformation regularization terms
        if self.use_deformation_mlp and cam_idxes >0:
            L_def_mag = (self.offsets ** 2).mean()
            L_frame = (self.frame_codes.weight ** 2).mean()
            #Log motion statistics
            offset_norm = self.offsets.norm(dim=-1)              # [N_up]
            max_offset = offset_norm.max()
            mean_offset = offset_norm.mean().item()
            max_offset=max_offset.item()
        else:
            L_frame=torch.zeros(1).cuda()
            L_def_mag=torch.zeros(1).cuda()
        

    

        if self.gaus_trainer.baby_hair_weight>1 :
              self.gaus_mask=self.gaus_mask/self.gaus_trainer.baby_hair_weight
           

        losses.update({
                'gaus_mask': mask,
                'gaus_orient': orient,
                'smoothnes_length_neighbor': L_length_neighbor,
                'smoothnes_dir_neighbor': L_dir_neighbor,
                'smoothnes_length_back': back_L_length ,

                'gaus_l1': l1,
                'gaus_ssim': ssim,
                'gaus_depth': depth,
                'pca_prior' :  self.L2_pca_prior_loss,
                'pca_reg': self.L2_pca_reg,
                'pca_reg_front_back': self.L2_pca_reg_front_back,
                'canonic_reg': self.L_canonic,
                'motion_reg' : L_def_mag,
                'frame_reg' : L_frame,

                'per_region_length':self.per_region_loss ,
                'gravity':L_gravity ,

            })
        
        # SDF loss
        if self.sdf_penalty > 0:
            head_dists, head_normals = self.sdf_head.query(world_strands[0].reshape(-1, 3))
            dists = head_dists.reshape(-1)
            losses['sdf'] = torch.relu( -dists).abs().mean()


        loss = (
                self.sdf_penalty * losses['sdf'] +
                self.gaus_mask * losses['gaus_mask'] +
                self.gaus_orient * losses['gaus_orient'] +
                
                self.smoothnes_length_neighbor*losses['smoothnes_length_neighbor']+
                self.smoothnes_dir_neighbor*losses['smoothnes_dir_neighbor']+
                self.smoothnes_length_back*losses['smoothnes_length_back']+
                self.smoothnes_dir_back*losses['smoothnes_dir_back']+

                self.pca_reg*losses['pca_reg']+
                self.pca_prior_reg *losses['pca_prior']+
                self.canonic_reg *losses['canonic_reg']+
                self.pca_reg_front_back*losses['pca_reg_front_back']+
                
                self.motion_reg*losses['motion_reg']+
                self.lambda_frame*losses['frame_reg']+

                self.gaus_depth * losses['gaus_depth'] +
            
                self.gaus_l1 * losses['gaus_l1'] + #0
                self.gaus_ssim * losses['gaus_ssim'] + #0
                self.gravity*losses['gravity'] 
                #self.region_length_loss * losses['per_region_length'] 
            )
       

#         print(losses)
        if self.step % 1 == 0 and self.use_deformation_mlp and cam_idxes >0:
            
            tip_offsets = self.offsets[:, -20:, :]   # last 20 points
            tip_norm = tip_offsets.norm(dim=-1)
            frame_code_norm = self.frame_codes.weight.norm(dim=-1)

            print(
                self.step,
                "data_loss:", loss.item(),
                "mean_offset:", mean_offset ,
                "max_offset:", max_offset ,
                "tip mean:", tip_norm.mean().item(), "tip max:", tip_norm.max().item(),
                "frame_code_mean_norm:", frame_code_norm.mean().item(),
            )

        

        # Save point clouds
        if self.step % self.pc_freq == 0 and rank == 0:
            #save_path = os.path.join(self.savedir, f"pointclouds_{mode}", f'pred_{self.step:06d}.ply')
            #if self.use_deformation_mlp :
            cam_idx = cam_idxes.item()
            save_path = os.path.join(self.savedir, f"pointclouds_{mode}", f'pred_{self.step:06d}_cam_idx{cam_idx:02d}.ply')
         
            save_strands(world_strands[0], save_path, num_points=self.num_points, cols=self.colors_save)

            if self.step == 0 : #or self.step== (self.all_steps-1):
               save_path = os.path.join(self.savedir, f"pointclouds_{mode}", f'pred_canonical_{self.step:06d}.ply')
               save_strands(pred_points_vis[0], save_path, num_points=self.num_points, cols=self.colors_save)

        # Log losses
        logs = {f'bs_{mode}': 1, f'full_loss_{mode}': loss.detach().cpu().numpy()}
        for k, v in losses.items():
            #print(v)
            if not v==0:
               #print(k)
               logs[f'loss_{k}_{mode}'] = v.detach().cpu().numpy()
        logs[f'baby_hair_weight_{mode}'] = self.gaus_trainer.baby_hair_weight
        logs[f'scale_head'] = self.gaus_trainer.scale_head
        if self.use_deformation_mlp and cam_idxes >0:
            logs[f'max_offset'] = max_offset
            logs[f'mean_offset'] = mean_offset

        #log weights
        if self.step == 0:
            hparams = {
                "sdf_penalty": float(self.sdf_penalty),
                "gaus_l1": float(self.gaus_l1),
                "gaus_ssim": float(self.gaus_ssim),
                "gaus_mask": float(self.gaus_mask),
                "gaus_orient": float(self.gaus_orient),
                "gaus_depth": float(self.gaus_depth),
                "smoothnes_length_neighbor": float(self.smoothnes_length_neighbor),
                "smoothnes_dir_neighbor": float(self.smoothnes_dir_neighbor),
                "smoothnes_dir_back": float(self.smoothnes_dir_back),
                "smoothnes_length_back": float(self.smoothnes_length_back),

                "region_length_loss": float(self.region_length_loss),
                "pca_prior_reg": float(self.pca_prior_reg),
                "pca_reg": float(self.pca_reg),
                "pca_reg_front_back": float(self.pca_reg_front_back),
                "gravity": float(self.gravity),
                "motion_reg": float(self.motion_reg),
                'lambda_frame': float(self.lambda_frame),                
            }
            text_block = json.dumps(hparams, indent=2)
            self.writer.add_text("hyperparameters", f"```\n{text_block}\n```", 0)
        return loss, logs
   


    def update_hairstyle(self, batch,  world_size, rank, device, global_rank):
        n_unfreeze_comp = max(5, min(self.step // self.unfreeze_time_for_pca, self.num_components)) if self.unfreeze_time_for_pca > -1 else self.num_components
    
        if self.step <= self.num_steps_coarse and self.unfreeze_time_for_pca==-1:
           n_unfreeze_comp=10 #self.num_components
 
        img, baldness_mask,  feats, cam, flip, transformer_mask, cam_idxes, gaus_feats_frontal,  gaus_cam_frontal ,sample_idx= batch


        text_condition = None    
        
        img, baldness_mask,  cam, flip, transformer_mask, cam_idxes, gaus_feats_frontal,  gaus_cam_frontal ,sample_idx= img.to(self.device), baldness_mask.to(self.device), cam.to(self.device), flip.to(self.device), transformer_mask.to(self.device), cam_idxes.to(self.device), gaus_feats_frontal.to(self.device),  gaus_cam_frontal.to(self.device), sample_idx.to(self.device)
        
        img, baldness_mask,  cam, flip, transformer_mask, cam_idxes, gaus_feats_frontal,  gaus_cam_frontal,sample_idx = img.to(rank), baldness_mask.to(rank),  cam.to(rank), flip.to(rank), transformer_mask.to(rank), cam_idxes.to(rank), gaus_feats_frontal.to(rank), gaus_cam_frontal.to(rank) ,sample_idx.to(rank)

        feats = feats.to(self.device)
        feats = feats.to(rank)

        model_input = img

        transformer_mask_cond = -100 * (1 - (F.interpolate(transformer_mask.unsqueeze(1), (self.transformer_mask_size, self.transformer_mask_size)).reshape(model_input.shape[0], -1).unsqueeze(1).unsqueeze(2) > 0).float())
    

        transformer_mask =  F.interpolate(transformer_mask.unsqueeze(1), (self.transformer_mask_size, self.transformer_mask_size), mode='bilinear', align_corners=False)

        if self.dilate_mask:

            transformer_mask = F.max_pool2d(transformer_mask, kernel_size=3, stride=1, padding=1).squeeze(1)

        transformer_mask_cond = -100 * (1 - transformer_mask.reshape(model_input.shape[0], -1).unsqueeze(1).unsqueeze(2) > 0)

        # #forward pass
        # if self.step==0:
        #         print('finetune coarse model')
        #         self.lp_enc_elow.train()
        # if self.step>2:
        #         print('finetune fine model')
        #         self.lp_enc_elow.eval()
        batched_pred_strand_dirs, batched_pred_scaling_factor = self.lp_enc(model_input, text_condition, camera_token=None, transformer_mask=transformer_mask_cond, elo=self.lp_enc_elow)
        
        #######
        strand_dirs = batched_pred_strand_dirs * self.scale_stats_std.reshape(1, -1, 1, 1) + self.scale_stats_mean.reshape(1, -1, 1, 1)            
        pred_strands_dirs = strand_dirs.permute(0, 2, 3, 1).reshape(-1, 64)
        #torch.save(pred_strands_dirs,'/home/ayed/prerocessing_input_monocular_hair_modelling/person140/prior_pca.pt')
        if self.step ==0:
            self.z_prior=pred_strands_dirs#.detach()
            prior_path=self.config['loss_config'].get('pca_prior_path', 'None')
            if prior_path!='None':
                self.z_prior=torch.load(prior_path).to(device)       
                    
            with torch.no_grad():
                self.scalp_map64= torch.tensor(np.array(Image.open(self.scalpPartionningPath64x64))).cuda().detach()/255
        current_z=pred_strands_dirs#.detach()
        #Regularization Strategy for Unseen PCA Values
        if self.pca_prior_reg >0 :
            self.L2_pca_prior_loss = self.prior_pca_regularization_loss(current_z, self.scalp_map64, baldness_mask)
        else:
            self.L2_pca_prior_loss=torch.zeros(1).cuda()
        
        if self.pca_reg_front_back >0 :
           self.L2_pca_reg_front_back=self.pca_regularization_loss(current_z, self.scalp_map64) #baldness_mask
        else:
           self.L2_pca_reg_front_back=torch.zeros(1).cuda()

        if self.pca_reg>0:
            self.L2_pca_reg=self.pca_l2(current_z)
        else:
            self.L2_pca_reg=torch.zeros(1).cuda()
        self.L_canonic=torch.zeros(1).cuda()


        #######


        batched_pred_strand_dirs = batched_pred_strand_dirs[:, :n_unfreeze_comp]
        if self.scale_output:
            #unnormalize coeffs with dataset stats
            batched_pred_strand_dirs = batched_pred_strand_dirs * self.scale_stats_std[:n_unfreeze_comp].reshape(1, -1, 1, 1) + self.scale_stats_mean[:n_unfreeze_comp].reshape(1, -1, 1, 1)            
            
            
        if batched_pred_scaling_factor.shape[1] > 1:
            batched_pred_scaling_factor, batched_pred_baldness_mask = torch.split(batched_pred_scaling_factor, 1, dim=1)
        
                
        if self.use_scale is False:
            batched_pred_scaling_factor = torch.ones_like(batched_pred_scaling_factor)

        bs = img.shape[0]
                
        if self.edited_uvmap is None:
            edit_uvmap_path=self.config['gaussians'].get('edit_uvmap', 'None')
            if edit_uvmap_path is None:

                gt_hair_mask_for_scalp = (((gaus_feats_frontal[0][0][3]>0)*255).detach().cpu().numpy()).astype(np.uint8)
                kernel = np.ones((5, 5), np.uint8)  # Size of kernel (5x5 in this case)
                # TODO adapt kernel check gt_hair_mask_dilated on my mask 
                dilated_mask = cv2.dilate(gt_hair_mask_for_scalp, kernel, iterations=1)
                gt_hair_mask_dilated = ((1 - torch.tensor(dilated_mask, device=device) / 255.) > 0).bool()

                self.scalp_render_map = self.scalp_render(gaus_cam_frontal[0][0],  gt_hair_mask_dilated)[None][None]
                self.edited_uvmap  = ((1 - torch.nn.functional.interpolate(self.scalp_render_map, (self.resolution_upsample, self.resolution_upsample),  mode='bilinear')) > 0.)[0][0]           
                #torch.save(self.edited_uvmap,'edited_uvmap.pt')
            else:
                self.edited_uvmap= torch.load(edit_uvmap_path)


        if self.scalp_mask_prediction is None:
            self.scalp_mask_prediction = batched_pred_baldness_mask.detach()  

        pred_strands_dirs = batched_pred_strand_dirs.permute(0, 2, 3, 1).reshape(-1, n_unfreeze_comp)
        pred_scaling_factor = batched_pred_scaling_factor.permute(0, 2, 3, 1).reshape(-1, 1)

        pred_pc = decode_pca(pred_strands_dirs, self.mean_shape,  self.blend_shapes, n_components=n_unfreeze_comp, num_points=self.num_points) * pred_scaling_factor.view(-1, 1, 1)

        roots = self.roots_origins.repeat(bs, 1, 1, 1).reshape(-1, 1, 3)
        
        strands_number =  self.texture_size ** 2
        
        pred_points_vis = torch.cat((roots, pred_pc  + roots), 1).reshape(bs,  strands_number, -1, 3) #[bs,  HW, num_pts,3] 
        
        if self.upsample_hairstyle:
#             print('upsample hairstyle')
            bs, hw, pts, ch = pred_points_vis.shape
            # Permute to bring spatial dimensions to PyTorch's expected order: (batch, channels, height, width)
            strand_texture = pred_points_vis.permute(0, 2, 3, 1).reshape(bs, pts*ch, 64, 64)
            
            # Upsample using haar interpolation
            pred_points_vis_local = pred_points_vis - pred_points_vis[:, :, :1]
            strand_texture = pred_points_vis_local[:, :, 1:].permute(0, 2, 3, 1).reshape(bs, -1, 64, 64) #597, 64, 64
            
            bil = F.interpolate(strand_texture, size=(self.resolution_upsample, self.resolution_upsample), mode='bilinear')[0]
            near = F.interpolate(strand_texture, size=(self.resolution_upsample, self.resolution_upsample), mode='nearest')[0] 

            nonzerox, nonzeroy = torch.where(baldness_mask[0][0] != 0)
            
            patch_world_displ = torch.zeros(64, 64, 199, 3, device=self.device)
            patch_world_displ[[nonzerox, nonzeroy]] = pred_points_vis_local.reshape(64, 64, 200, 3)[nonzerox, nonzeroy][:, 1:] - pred_points_vis_local.reshape(64, 64, 200, 3)[nonzerox, nonzeroy][:, :-1]
            
            strands_sim = calc_strands_similarity(patch_world_displ)
            strands_sim_hr = F.interpolate(strands_sim[None][None],size=(self.resolution_upsample, self.resolution_upsample), mode='bilinear')[0][0]
            
            
#                 print('hfhhdfh', near.shape, bil.shape)
            latents_interp = self.blend_func(strands_sim_hr)[None] * near + (1 - self.blend_func(strands_sim_hr)[None]) * bil
            pres = latents_interp.reshape(-1, 3, self.resolution_upsample, self.resolution_upsample).permute(2, 3, 0, 1).reshape(1, self.resolution_upsample * self.resolution_upsample, -1, 3)

            upsampled_baldness_mask = (F.interpolate(baldness_mask, size=(self.resolution_upsample, self.resolution_upsample), mode='bilinear', align_corners=False) >0.9)* self.edited_uvmap [None][None]
            ## TODO : uncomment the multiplication  
            
            interested_idxes_up = torch.where(upsampled_baldness_mask[0].reshape(-1) >=0.99)[0]

            if self.use_deformation_mlp and cam_idxes >0:
                ##      predict deformation offsets     ###
                upsampled_texture_canonic = torch.cat((self.roots_origins_up.reshape(1, -1, 1, 3), self.roots_origins_up.reshape(1, -1, 1, 3)+pres), -2)

                frame_code_k = self.get_frame_code(sample_idx-1)
                pres_deformed, offsets = self.deformerMLP(
                    pres=pres,
                    interested_idxes_up=interested_idxes_up,
                    frame_code=frame_code_k,
                    step=self.step ,
                    warmup_steps=self.canonical_only_steps,
                )

                roots_origins_up=self.roots_origins_up.reshape(1, -1, 1, 3)
                N_texels = pres.shape[1]
                upsampled_texture_deformed = torch.cat(
                    (
                        roots_origins_up.reshape(1, N_texels, 1, 3),
                        roots_origins_up.reshape(1, N_texels, 1, 3) + pres_deformed.unsqueeze(0)
                    ),
                    dim=-2
                )  # [1, N_texels, 200, 3]
                upsampled_texture = upsampled_texture_deformed 
                self.offsets=offsets
                self.debug_strands_with_most_motion(upsampled_texture_deformed,
                    upsampled_texture_canonic,
                    interested_idxes_up,
                    sample_idx
                )


                #For non‑canonical frames, penalize deviation from canonical
                pred_current=upsampled_texture_canonic[0][interested_idxes_up].detach()
                if self.canonic_strands is not None :
                    self.L_canonic=((pred_current - self.canonic_strands )**2).mean()


            else:
                upsampled_texture = torch.cat((self.roots_origins_up.reshape(1, -1, 1, 3), self.roots_origins_up.reshape(1, -1, 1, 3)+pres), -2)
            #upsampled_texture = torch.cat((self.roots_origins_up.reshape(1, -1, 1, 3), self.roots_origins_up.reshape(1, -1, 1, 3)+pres), -2)
            self.upsampled_texture=upsampled_texture
            if cam_idxes==0:
                self.canonic_strands = upsampled_texture[0][interested_idxes_up].detach()
            
            
        nonzero_idxs = torch.where(baldness_mask.reshape(-1) > 0)[0]

        if self.upsample_hairstyle:
            self.interested_idxes_up=interested_idxes_up

            
        else:
            self.appearance = None
            param_size = (interested_idxes_up.shape[0], self.num_points-1, 48)
            self.appearance = nn.Parameter(torch.ones(param_size, device=self.device)[nonzero_idxs].detach().contiguous().clone(), requires_grad=True)
       
        if self.upsample_hairstyle:
            selected_strands = upsampled_texture[0][interested_idxes_up]
            diffs = (upsampled_texture[0, :, 1:, :] - upsampled_texture[0, :, :-1, :]) # [strands_number, num_pts-1, 3]
            segment_lengths = torch.norm(diffs, dim=2)                             # (N, P-1)
            strand_lengths = segment_lengths.sum(dim=1)                       # [strands_number, num_pts-1]
            scale_to_cm = self.scale_can2world * 100.0 
            strand_lengths_cm = strand_lengths * scale_to_cm
            length_map = strand_lengths_cm.view(256, 256)
            baldness_mask=upsampled_baldness_mask


        else:
            selected_strands = pred_points_vis[0][nonzero_idxs]
            diffs = (pred_points_vis[0, :, 1:, :] - pred_points_vis[0, :, :-1, :]) #[strands_number, num_pts-1, 3]
            segment_lengths = torch.norm(diffs, dim=2)                             # (N, P-1)
            strand_lengths = segment_lengths.sum(dim=1)  
            scale_to_cm = self.scale_can2world * 100.0 
            strand_lengths_cm = strand_lengths * scale_to_cm 
            length_map = strand_lengths_cm.view(64, 64)
        

        #save roots
        if self.step==0:
                pcd = o3d.geometry.PointCloud()
                roots_np = selected_strands[:, 0, :].detach().cpu().numpy()
                pcd.points = o3d.utility.Vector3dVector(roots_np)
                o3d.io.write_point_cloud(f'{self.savedir}roots.ply', pcd, write_ascii=True)
                torch.save(selected_strands,f"{self.savedir}selected_strands.pt")
                torch.save(interested_idxes_up,f"{self.savedir}interested_idxes_up.pt")
        return selected_strands[None]

    def debug_strands_with_most_motion(self, upsampled_texture_deformed, upsampled_texture_canonic, interested_idxes_up, sample_idx):     
        
        if self.step %100 ==0:
            k=250
            offset_norm = self.offsets.norm(dim=-1)
            strand_motion = offset_norm.max(dim=1).values 
            top_vals, top_idx = torch.topk(strand_motion, k)       
            top_texel_ids = interested_idxes_up[top_idx]        # [k]
            # their deformed 3D strands (in upsampled_texture_deformed)
            top_strands_def = upsampled_texture_deformed[0][top_texel_ids]  # [k, 200, 3]
            top_strands_can = upsampled_texture_canonic[0][top_texel_ids]
            pcd1,pcd2 = o3d.geometry.PointCloud() , o3d.geometry.PointCloud()
            top_strands_def_np=top_strands_def.reshape(-1, 3).detach().cpu().numpy()
            top_strands_can_np=top_strands_can.reshape(-1, 3).detach().cpu().numpy()
            pcd1.points = o3d.utility.Vector3dVector(top_strands_def_np)
            pcd2.points = o3d.utility.Vector3dVector(top_strands_can_np)

            os.makedirs(os.path.join(self.savedir, 'motion'), exist_ok=True)
            o3d.io.write_point_cloud(f'{self.savedir}motion/top_strands_deformed{self.step}.ply', pcd1, write_ascii=True)
            o3d.io.write_point_cloud(f'{self.savedir}motion/top_strands_can{self.step}.ply', pcd2, write_ascii=True)

            #save canonic_strands
            pred_points_vis = upsampled_texture_canonic[0][interested_idxes_up]
            n_strands = pred_points_vis.shape[0]
            n_pts = pred_points_vis.shape[1]            
            world_strands = (self.scale_can2world * (self.R_can2world @ pred_points_vis.reshape(-1, 3).T).T + self.t_can2world).reshape(1, n_strands, n_pts, 3)
            self.hairstyle_init = world_strands[0].reshape(-1, 200, 3).detach()
            self.colors_save = torch.cat((torch.rand(world_strands[0].shape[0], 3).unsqueeze(1).repeat(1, self.num_points, 1), torch.ones(world_strands[0].shape[0], self.num_points, 1)), dim=-1).reshape(-1, 4).cpu()
             
            save_path = os.path.join(self.savedir, f"pointclouds_train", f'pred_{self.step:06d}_noOffsets.ply')
            save_strands(world_strands[0], save_path, num_points=n_pts, cols=self.colors_save)
        
    def transfrom_pcdW(self,cam_idxes):
        if self.step ==0:
            path=f'{self.savedir}roots.ply'
            hair_roots = o3d.io.read_point_cloud(path)
            hair_roots_pts=torch.from_numpy(np.array(hair_roots.points)).float().cuda()
            
            rootsW = self.scale_can2world * (self.R_can2world @ hair_roots_pts.T).T + self.t_can2world

            pcd = o3d.geometry.PointCloud()
            roots_npW = rootsW.detach().cpu().numpy()
            pcd.points = o3d.utility.Vector3dVector(roots_npW)
            o3d.io.write_point_cloud(f'{self.savedir}rootsW.ply', pcd, write_ascii=True)
        
        if self.use_deformation_mlp and cam_idxes>0:
            if self.step %100 ==0:

                path=f'{self.savedir}motion/top_strands_can{self.step}.ply'
                hair_strands = o3d.io.read_point_cloud(path)
                hair_strands_pts=torch.from_numpy(np.array(hair_strands.points)).float().cuda()
                strandsW = self.scale_can2world * (self.R_can2world @ hair_strands_pts.T).T + self.t_can2world
                pcd = o3d.geometry.PointCloud()
                strands_npW = strandsW.detach().cpu().numpy()
                pcd.points = o3d.utility.Vector3dVector(strands_npW)
                o3d.io.write_point_cloud(f'{self.savedir}motion/top_strands_can{self.step}.ply', pcd, write_ascii=True)
                
                path=f'{self.savedir}motion/top_strands_deformed{self.step}.ply'
                hair_strands = o3d.io.read_point_cloud(path)
                hair_strands_pts=torch.from_numpy(np.array(hair_strands.points)).float().cuda()
                strandsW = self.scale_can2world * (self.R_can2world @ hair_strands_pts.T).T + self.t_can2world
                pcd2 = o3d.geometry.PointCloud()
                strands_npW = strandsW.detach().cpu().numpy()
                pcd2.points = o3d.utility.Vector3dVector(strands_npW)
                o3d.io.write_point_cloud(f'{self.savedir}motion/top_strands_deformed{self.step}.ply', pcd2, write_ascii=True)


    def log_statistics(self, stats , baldness_mask , Partition_ID_Map, length_map):
        if self.step %50 ==0: 
            if self.step==0:
                torch.save(Partition_ID_Map, f"{self.savedir}Partition_ID_Map.pt")

            self.log_length_maps( baldness_mask, length_map  )
            valid_mask = (length_map > 0)
            valid_lengths = length_map[valid_mask]
            mean_len = valid_lengths.mean()
            std_len = valid_lengths.std()
            stats_text = f"Total Mean: {mean_len.item():.4f} cm\nStd: {std_len.item():.4f} cm\n"
            region_stats_text = f"Mean back: {stats['mean_back']:.4f} cm\nMean sides: {stats['mean_sides']:.4f} cm\nMean_middle: {stats['mean_middle']:.4f} cm\n"
            with open(f"{self.savedir}lengthmaps/hair_length_stats{self.step}.txt", "w") as f:
                f.write(stats_text)
                f.write('-----\n')
                f.write(region_stats_text)


    def prepare_needed_data_for_region_scalp(self, gaus_cam_frontal, gaus_feats_frontal,baldness_mask ,  pred_points_vis ):
        

        edit_uvmap_path=self.config['gaussians'].get('edit_uvmap', 'None')
        if edit_uvmap_path is None:

            gt_hair_mask_for_scalp = (((gaus_feats_frontal[0][0][3]>0)*255).detach().cpu().numpy()).astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)  # Size of kernel (5x5 in this case)
            dilated_mask = cv2.dilate(gt_hair_mask_for_scalp, kernel, iterations=1)
            gt_hair_mask_dilated = ((1 - torch.tensor(dilated_mask, device=baldness_mask.device) / 255.) > 0).bool()

            self.scalp_render_map = self.scalp_render(gaus_cam_frontal[0][0],  gt_hair_mask_dilated)[None][None]
            self.edited_uvmap = ((1 - torch.nn.functional.interpolate(self.scalp_render_map, (self.resolution_upsample, self.resolution_upsample),  mode='bilinear')) > 0)[0][0]           
            self.edited_uvmap=torch.load('/home/ayed/3Dhairstyle/inputs/example_data72/edited_uvmap.pt')
        
        else:
            self.edited_uvmap= torch.load(edit_uvmap_path)
            upsampled_baldness_mask = (F.interpolate(baldness_mask, size=(self.resolution_upsample, self.resolution_upsample), mode='bilinear', align_corners=False)>0.9) * self.edited_uvmap [None][None]
            interested_idxes_up = torch.where(upsampled_baldness_mask[0].reshape(-1) >=0.99)[0]
 
        # # region length
        # B,_, npoints,_ =pred_points_vis.shape
        # upsampled_texture = torch.zeros((1, self.resolution_upsample*self.resolution_upsample,npoints,3)).to(baldness_mask).float()
        # upsampled_texture[0][interested_idxes_up] = pred_points_vis
        # diffs = (upsampled_texture[0, :, 1:, :] - upsampled_texture[0, :, :-1, :]) # [strands_number, num_pts-1, 3]
        # segment_lengths = torch.norm(diffs, dim=2)                             # (N, P-1)
        # strand_lengths = segment_lengths.sum(dim=1)                       # [strands_number, num_pts-1]
        # scale_to_cm = self.scale_can2world * 100.0 
        # strand_lengths_cm = strand_lengths * scale_to_cm
        # length_map = strand_lengths_cm.view(256, 256)
        length_map=None
        hair_mask = (upsampled_baldness_mask[0,0]>0 ).float()
        return length_map , interested_idxes_up, hair_mask



    def per_region_length_loss_soft(self, length_map_cm, influence_map_3d, 
                                hair_mask, means_cm, stds_cm, lows_cm, highs_cm, 
                                w_gauss, w_bounds):
        """
        Calculates regional length loss using soft influence maps (normalized_weight_maps).
        
        Args:
            length_map_cm (H,W): Tensor of generated strand lengths.
            influence_map_3d (H,W,N): Soft weight map tensor (N channels, e.g., N=5).
            ... other parameters ...
        """
        H, W = length_map_cm.shape
        device = length_map_cm.device

        # 1. Prepare Inputs: Consolidate N (e.g., 5) influence channels into 3 logical regions
        # NOTE: This assumes influence_map_3d is (H, W, 5) and we are mapping IDs (1, 2, 3, 4, 5).
        
        # We must assume the 5-channel influence map has been consolidated into a 3-channel (Front, Back, Side)
        # tensor W before being passed or must be done inside the function.
        
        # --- LOGICAL MASK CONSOLIDATION (Crucial Step: Must be adjusted if order changes) ---
        # We assume influence_map_3d is already structured or sliced to provide 3 channels (Sides, Back, Middle)
        # If the input tensor is (H, W, 5), you must slice and combine channels 
        # to create a consolidated (H, W, 3) tensor W_consolidated.
        
        # Example for 5 channels -> 3 logical regions:
        # (Assuming indices 0, 3 are Sides; 4 is Back; 1, 2 are Middle)
        W_sides = influence_map_3d[..., 0] + influence_map_3d[..., 1] 
        W_back = influence_map_3d[..., 3]
        W_middle = influence_map_3d[..., 2] #+ influence_map_3d[..., 2]
        W_consolidated = torch.stack([W_sides, W_back, W_middle], dim=-1) # (H, W, 3)
        
        # Permute to (3, H, W) for efficient broadcast multiplication
        W = W_consolidated.permute(2, 0, 1).to(device)  # W is now (3, H, W)
        
        # GT parameters must be viewed as (3, 1, 1)
        mu = torch.tensor(means_cm, device=device).view(3, 1, 1)
        sd = torch.tensor(stds_cm, device=device).view(3, 1, 1).clamp_min(0.01)
        lo = torch.tensor(lows_cm, device=device).view(3, 1, 1)
        hi = torch.tensor(highs_cm, device=device).view(3, 1, 1)

        L = length_map_cm.unsqueeze(0)  # (1,H,W)

        # Calculate overall active pixel count for global normalization
        hair_mask_sum_active = hair_mask.sum().clamp_min(1.0) 

        # 2. Calculate Gaussian Loss (Softened)
        # z2 is the squared error from the mean for all 3 regions
        z2 = ((L - mu) / (sd))**2                     # (3, H, W)
        
        # (z2 * W) weights the squared error by the soft influence.
        gauss_per_px = (z2 * W).sum(0)                 # (H,W) - Sum of weighted contributions
        
        # Normalize by the total number of active hair pixels
        gauss_loss = (gauss_per_px * hair_mask).sum() / hair_mask_sum_active

        # 3. Calculate Soft Bounds Loss (Softened)
        # Using Softplus for stable hinge loss
        BETA = 10.0
        under = F.softplus(lo - L, beta=BETA) 
        over  = F.softplus(L - hi, beta=BETA)
        
        # (under + over) * W weights the bound violation by the influence (W)
        bounds_per_px = ((under + over) * W).sum(0) 
        bounds_loss = (bounds_per_px * hair_mask).sum() / hair_mask_sum_active

        total = w_gauss * gauss_loss + w_bounds * bounds_loss

        # quick stats for logging
        def masked_mean_soft(x, m):
            s = (x*m).sum(); n = m.sum().clamp_min(1.0); return (s/n).item()

        stats = {
            # Note: You'll need to pass m_back, m_sides, m_middle weights separately for logging
            "mean_sides": masked_mean_soft(length_map_cm, W_sides),
            "mean_back": masked_mean_soft(length_map_cm, W_back),
            "mean_middle": masked_mean_soft(length_map_cm, W_middle),
            "gauss_loss": gauss_loss.item(),
            "bounds_loss": bounds_loss.item(),
            "total": total.item(),
        }
        return total, stats

    def find_neighbors(self, selected_strands, K=5):
        """ Finds K nearest neighbors for each strand root. """
        roots = selected_strands[:,:, 0, :] # Shape (1, N, 3)
        
        # knn_points finds the K nearest neighbors of p1 in p2. Here p1 == p2.
        # idx contains the indices of the K nearest neighbors in the N total strands.
        # idx shape: (1, N, K)
        knn = knn_points(p1=roots, p2=roots, K=K+1) # K+1 to exclude the strand itself
        neighbor_indices = knn.idx[0, :, 1:] # Exclude the first index (which is the strand itself)
        return neighbor_indices # Shape (N, K)

    def neighbor_smoothness_loss(self, selected_strands,K=1):
        
        strands = selected_strands.squeeze(0)
        
        # 1. Calculate Segment Properties for ALL strands (i)
        # Dirs (N, S-1, 3); Magnitudes (N, S-1)
        dirs_i = strands[:, 1:] - strands[:, :-1]
        mags_i = torch.linalg.norm(dirs_i, dim=-1).clamp_min(1e-6)
        
        # Normalized directions (N, S-1, 3)
        D_unit_i = dirs_i / mags_i.unsqueeze(-1)

        neighbor_indices=self.find_neighbors(selected_strands, K=K)

        # 2. Collect Neighbor Properties (j) using neighbor_indices (N, K)
        # Gather neighbor directions and magnitudes
        # D_unit_j shape: (N, K, S-1, 3)
        D_unit_j = D_unit_i[neighbor_indices] 
        mags_j = mags_i[neighbor_indices] # (N, K, S-1)

        # 3. Calculate Average Neighbor Properties
        # Avg_D_unit_j shape: (N, S-1, 3)
        Avg_D_unit_j = D_unit_j.mean(dim=1)
        Avg_mags_j = mags_j.mean(dim=1) # (N, S-1)

        # --- Length Smoothness Component (L2 Penalty) ---
        # Penalize (Mag_i - Avg_Mag_j)^2
        length_diff = mags_i - Avg_mags_j
        L_length_neighbor = length_diff.pow(2).mean()

        # --- Direction Smoothness Component (1 - Cosine Similarity Penalty) ---
        # Clamp cosine similarity to prevent numerical issues
        cos_sim = (D_unit_i * Avg_D_unit_j).sum(dim=-1).clamp(min=-1.0, max=1.0)
        L_dir_neighbor = (1.0 - cos_sim).mean()
        return L_dir_neighbor , L_length_neighbor
    

    def back_smoothness_loss(self, selected_strands, Partition_ID_Map, interested_idxes_up):
        strands = selected_strands.squeeze(0)  # [N, S, 3]
        neighbor_indices = self.find_neighbors(selected_strands, K=1)

        segs = strands[:, 1:] - strands[:, :-1]                    # [N, S-1, 3]
        seg_lens = torch.linalg.norm(segs, dim=-1).clamp_min(1e-6) # [N, S-1]
        total_len = seg_lens.sum(dim=-1)                            # [N]

        total_len_j = total_len[neighbor_indices]                   # [N, K]
        avg_total_len_j = total_len_j.mean(dim=1)                   # [N]

        L_len = (total_len - avg_total_len_j).pow(2)                # [N]

        back_mask_flat = (Partition_ID_Map == 2.0).reshape(-1)
        back_strands_mask = back_mask_flat[interested_idxes_up]

        return L_len[back_strands_mask].mean() 
            
    
    def pca_regularization_loss(self, pred_strands_dirs, scalp_map64, ):
        """
        Calculates L1 loss between the mean PCA coefficient vector of the front (target) 
        and the mean PCA coefficient vector of 
        """
        back_mask = (scalp_map64.flatten() == 1)
        front_mask = (scalp_map64.flatten() == 0) # this is working

        coeff_nb=pred_strands_dirs.shape[1]
        # C_front_PCA shape: (N_front, 64) - All PCA vectors in the front region
        C_front_PCA = pred_strands_dirs[back_mask]
        C_back_PCA = pred_strands_dirs[front_mask]

        N_front = C_front_PCA.shape[0]
        N_back = C_back_PCA.shape[0]
      
        # Calculate the mean vector across all strands in the front
        P_target = C_front_PCA.mean(dim=0) # Shape: (64,) - This is the vector target
        P_mean_back = C_back_PCA.mean(dim=0) # Shape: (64,) - This is the generated vector

        # --- 3. Calculate L1 Regularization Loss ---
        # The loss is the L1 distance between the 64-dimensional mean vectors.
        L_pca_reg_vector = torch.abs(P_mean_back - P_target)
        
        # Sum the L1 loss over all 64 dimensions
        L_pca_reg = L_pca_reg_vector.sum()
        return L_pca_reg

    def pca_l2(self, pred_strands_dirs):
        """
        L2 regularization that keeps PCA coefficients small (near zero).
        """
        L_pca_l2 = torch.sum(pred_strands_dirs ** 2)
        return L_pca_l2

        
    def compute_gravity_loss(self, strands, K_segments=15):
        """
        Gravity loss encouraging the last K segments of ALL strands
        to point downward (-Y direction).
        
        selected_strands: (1, N, S, 3)
        """

        # selected_strands: (1, N, S, 3)
        B, N, S, _ = strands.shape
        dirs = strands[:, :, 1:] - strands[:, :, :-1]

        # Normalize  
        magnitudes = torch.linalg.norm(dirs, dim=-1, keepdim=True)
        D_unit = dirs / magnitudes.clamp_min(1e-6)   # (1, N, S-1, 3)
        
        D_unit_last = D_unit[:, :, -K_segments:]               # (1, N, K, 3)
        V_gravity = torch.tensor([0.0, -1.0, 0.0], 
                                device=strands.device,
                                dtype=strands.dtype).view(1, 1, 1, 3)

        cos_alignment = (D_unit_last * V_gravity).sum(dim=-1)     # (1, N, K)
        cos_alignment = cos_alignment.clamp(-1.0, 1.0)
        L_gravity = (1.0 - cos_alignment).mean()
        return L_gravity
 

    def log_length_maps(self, baldness_mask, length_map):
        baldness_mask_np=baldness_mask.cpu().numpy()
        baldness_mask_normalized = cv2.normalize(baldness_mask_np, None, 0, 255, cv2.NORM_MINMAX)
        baldness_mask_uint8 = baldness_mask_normalized.astype(np.uint8)
        masked_length_map = np.where(baldness_mask_np > 0, length_map.detach().cpu().numpy(), 0)
        os.makedirs(os.path.join(self.savedir, 'lengthmaps'), exist_ok=True)
        vis = torch.from_numpy(masked_length_map)
        vmax = torch.quantile(vis[vis>0], 0.98) if (vis>0).any() else 1.0
        vmin = 0.0
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.imshow(vis, cmap='magma', vmin=vmin, vmax=float(vmax))
        plt.title('Strand length map'); plt.axis('off'); plt.colorbar(label='Strand Length (cm)')
        plt.savefig(f"{self.savedir}lengthmaps/length_map{self.step}.png", bbox_inches='tight', dpi=300)
        plt.close()
    
        if self.step%100==0:
            cv2.imwrite(f"{self.savedir}lengthmaps/baldness_mask{self.step}.png", baldness_mask_uint8)
            

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

          
    @torch.no_grad()
    def load_model(self, ckpt_path, rank):
        local_rank = dist.get_local_rank()
        print(f'Loading model on GPU {local_rank}')

        # Map the checkpoint to the current device
        map_location = {f'cuda:0': f'cuda:{rank}'}
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        print(f'Loaded checkpoint: {ckpt_path}')
        self.ckpt_path=ckpt_path

        state_dict = checkpoint['lp_enc']
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
                new_key = k.replace('module.', '')  # Remove `module.` prefix
                new_state_dict[new_key] = v
        self.lp_enc.load_state_dict(new_state_dict)

        # Attempt to load optional components
        try:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            state_dict = checkpoint['lp_enc_elow']
            for k, v in state_dict.items():
                    new_key = k.replace('module.', '')  # Remove `module.` prefix
                    new_state_dict[new_key] = v
            self.lp_enc_elow.load_state_dict(new_state_dict)
            #self.lp_enc_elow.load_state_dict(checkpoint['lp_enc_elow'])
            print('Loaded lp_enc_elow successfully')
        except Exception as e:
            print(f'Failed to load lp_enc_elow: {e}')

        try:
            self.optimizer=self.configure_optimizers()
            if not self.finetune_coarse_model:
                # 3) Prune optimizer state from the checkpoint and load it
                ckpt_opt = checkpoint['optimizer_state_dict']
                # Compute how many params your current optimizer actually has in its (first) group
                keep_n = len(self.optimizer.param_groups[0]['params'])   # 77 in your printout
                ckpt_opt_pruned = self.prune_optimizer_state_to_n(ckpt_opt, keep_n)
                self.optimizer.load_state_dict(ckpt_opt_pruned)

            else:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded optimizer state')
        except Exception as e:
            print(f'Failed to load optimizer state: {e}')


    def prune_optimizer_state_to_n(self, ckpt_opt_state, keep_n):
        """
        Return a copy of `ckpt_opt_state` (from checkpoint['optimizer_state_dict'])
        that only contains the first `keep_n` parameters of each param group.
        Also removes any per-parameter state for the dropped params.

        Works for Adam/AdamW/etc. Assumes groups were formed by concatenating lists,
        so the first `keep_n` params correspond to the block you want to keep.
        """
        import copy

        pruned = copy.deepcopy(ckpt_opt_state)
        pruned_state = pruned["state"]
        pruned_groups = pruned["param_groups"]

        # Collect the param ids we keep across all groups (you have one group)
        keep_ids = set()
        for g in pruned_groups:
            old_ids = g["params"]
            if len(old_ids) < keep_n:
                raise ValueError(f"Checkpoint group has only {len(old_ids)} params, but keep_n={keep_n}.")
            g["params"] = old_ids[:keep_n]
            keep_ids.update(g["params"])

        # Drop optimizer state for removed params
        for pid in list(pruned_state.keys()):
            if pid not in keep_ids:
                del pruned_state[pid]

        return pruned

    @torch.no_grad()
    def save_model(self):
        local_rank = dist.get_local_rank()
        print(f'Saving model on GPU {local_rank}')

        checkpoint = {
            'lp_enc': self.lp_enc.state_dict(),
            'lp_enc_elow': self.lp_enc_elow.state_dict(),
            'step': self.step,
            'optimizer_state_dict': self.optimizer.state_dict()
        }

        checkpoint_dir = os.path.join(self.savedir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_{self.step:06d}.pth')
        torch.save(checkpoint, checkpoint_path)
    

    def train(self, world_size, rank, device, global_rank):
        try: 
          print('print(len(self.train_dl)',len(self.train_dl))
          with tqdm(total=self.all_steps, desc="Training steps", dynamic_ncols=True, smoothing=0.3) as pbar:

            while self.epoch < self.all_steps:  
             #while True:
                #print(f'Starting epoch {self.epoch} ')
                for batch in self.train_dl:
                    #print('batch',batch)
                    loss, logs = self.training_step(batch, self.step, world_size,rank, device, global_rank)
                    if self.accumulate_gradients > 1:
                        loss /= self.accumulate_gradients
                        
                    loss.backward()  
                    optimize_only_vis_strand=False #True
                    if optimize_only_vis_strand and self.scanner_type=='scanner_3d':  
                        self.config
                        hair_vis_mask_path=self.config['loss_config']['hair_vis_mask_path']
                        hair_vis_mask=torch.load(hair_vis_mask_path)
                        segment_vis_mask = torch.repeat_interleave(hair_vis_mask, self.num_points - 1)
                        # 2. Reshape to [N * 199, 1] for element-wise multiplication with your [N * 199, 3] dirs
                        segment_vis_mask = segment_vis_mask.unsqueeze(-1).float()
                        with torch.no_grad():
                            self.hairstyle_dirs.grad *= segment_vis_mask
                
                    if (self.step + 1) % self.accumulate_gradients == 0:

                        self.optimizer.step()
                        self.optimizer.zero_grad()
          
                    if self.step % self.logging_freq == 0 and rank == 0:
                        print('start logging')
                        for key in logs:
                            self.writer.add_scalar(f'{key}', logs[key], self.step)
                    
                    if self.step % self.save_freq == 0:# and rank == 0:
                        print('start saving')
                        self.save_model()
                        
                    if self.step ==0:
                        import json
                        hparams = {
                            "use_directed_loss": self.config['gaussians']['use_directed_loss'] ,
                            "convert_gabor_to_strand_map": self.config['dataset_real']['convert_gabor_to_strand_map'] ,
                            "use_SingleView": self.config['dataset_real']['use_SingleView'] ,
                            "use_orientations2": self.config['dataset_real']['use_orientations2'] ,
                            "use_improvedHairMask": self.config['dataset_real']['use_improvedHairMask'] ,
                            "use_deformation_mlp": self.config['dataset_real']['use_deformation_mlp'] ,
                        }
                        text_block = json.dumps(hparams, indent=2)
                        self.writer.add_text("datset_params", f"```\n{text_block}\n```", 0)


                    self.step += 1
                    pbar.update(1)
                    pbar.set_postfix({'loss': float(loss.item())})       
                  
                self.epoch += 1

                #if self.epoch > self.all_steps:
                #    break 
            if self.epoch > self.all_steps:
               pbar.close()
            exit(0) 
        except KeyboardInterrupt:
            pass
