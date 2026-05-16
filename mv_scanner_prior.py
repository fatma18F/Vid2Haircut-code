# === Standard Library ===
import os
import sys
import random
import argparse
import pickle

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
from src.utils.geometry import compute_similarity_transform, can2world_transform

from model_utils.get_projector import create_projector_backbone

# Arguments / config handling
from arguments import ModelParams, PipelineParams, OptimizationParams

# Distributed training utilities
from src.utils import distributed as dist

# === Environment and Torch Settings ===
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from mv_scanner import BaseScanTrainer

import trimesh
from PIL import Image

class ScanTrainer(BaseScanTrainer):
    def __init__(self,
                 config,
                 world_size,
                 rank, 
                 device,
                 global_rank,
                 ckpt_path=None,
                 savedir='./exps/vqvae_32x32x8_lr_1e-5_channels_122_attn_32_embed_dim_8_bs_16_disc_01',
                 unfreeze_time_for_pca=-1,
                 ngpus=-1,
                 num_workers=0,
                 accumulate_gradients=1,
                 dataset=None,
                 opt=None,
                 opt_hair=None,
                 pipe=None,
                 pointcloud_path_head=None, 
                 ip=None,
                 port=None,
                 prefix2='', 
                 scene='',
                 upsample_hairstyle=False, 
                 upsample_resolution=64,
                 num_steps_coarse=200,
                 optimize_appearance=True,
                 ):
        
        nn.Module.__init__(self)
      
        self._init_basic_config(num_steps_coarse, device, ngpus, accumulate_gradients,
                            upsample_hairstyle, upsample_resolution, optimize_appearance, config, 
                            unfreeze_time_for_pca)
        
        if self.use_unreal_sim:
           self._init_dataset_sim(config, scene, prefix2, world_size, rank, num_workers)
        else:
           self._init_dataset(config, scene, prefix2, world_size, rank, num_workers)

        self._init_roots_and_blend_shapes()

        self._init_config(config)
            
        self._init_encoders(config, device, rank)

        self._init_gaussian_trainer(dataset, opt, opt_hair, pipe, pointcloud_path_head, ip, port, rank, config)

        self._setup_dirs_and_writer(savedir)
        
        self._init_canonical_to_world(config)
        
        if self.use_deformation_mlp:
          self._init_deformation(config, device, rank)
        else:
          self.use_stage_deform=False


        if ckpt_path:
            print('Loading checkpoint...')
            self.load_model(ckpt_path, rank)
        

        if self.use_stage_deform:
            print(f'Using canonical-only stage for first {self.canonical_only_steps} steps. Deformation MLP will be frozen.')
            self.use_deformation_mlp = False
    
        self.optimizer = self.configure_optimizers()

        self._upload_sdf_prior(config)
        print('quit _upload_sdf_prior')

    
    def training_step(self, batch, batch_idx, world_size, rank, device, global_rank, mode='train'):
        
        if self.use_stage_deform:
            if  self.step == self.canonical_only_steps:
                self.use_deformation_mlp = True
                print(f'Past canonical-only stage. Deformation MLP will be optimized.')
                self.optimizer = self.configure_optimizers()
        
        self.pred_points_vis = self.update_hairstyle(batch, world_size, rank, device, global_rank)
        self.scanner_type='scanner_prior'

#         print('points', pred_points_vis.shape)
        loss, logs = self.single_step(self.pred_points_vis, batch, batch_idx, world_size, rank, device, global_rank)
        
        if self.step % self.save_freq == 0 and rank == 0:
            print('start saving')
            self.save_model()

        return loss, logs
        
       
    def configure_optimizers(self, coarse=False ):
        param_groups = []

        # 1) Z / prior space (lp_enc, maybe lp_enc_elow)
        z_params = list(self.lp_enc.parameters())
        if self.finetune_coarse_model:
            z_params += list(self.lp_enc_elow.parameters())
        if self.gaus_trainer.optimize_scale:
           z_params += [self.gaus_trainer.scale_head]
        
        if self.optimize_appearance :
            param_groups.append({
                "params": [self.appearance],
                "lr": self.learning_rate * 50.0,
                "weight_decay": 0.0,
            })
            param_groups.append({
                "params": [
                    self.gaus_trainer.gaussians._features_dc,
                    self.gaus_trainer.gaussians._features_rest,
                    #self.gaus_trainer.gaussians_hair._opacity,
                ],
                "lr": self.learning_rate *10,
                "weight_decay": 0, 
            })


        param_groups.append({
            "params": filter(lambda p: p.requires_grad, z_params),
            "lr": self.learning_rate ,         # e.g. 1e-4
            "weight_decay": self.weight_decay
        })

        # 2) Motion: deformation MLP
        if self.use_deformation_mlp:
            param_groups.append({
                "params": self.deformerMLP.parameters(),
                "lr": 1e-2,   #  5  
                "weight_decay": 0.0             
            })
            param_groups.append({
                "params": self.frame_codes.parameters(),
                "lr": 1e-3,
                "weight_decay": 0.0
            })

        if self.optimizer_type == 'adam':
            opt_ae = torch.optim.Adam(
                param_groups,
                lr=self.learning_rate,  
                betas=(0.5, 0.9)
            )
        elif self.optimizer_type == 'adamw':
            opt_ae = torch.optim.AdamW(
                param_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

        return opt_ae



    
    
def main(args, dataset, opt, opt_hair, pipe, pointcloud_path_head,  ip=None, port=None):


    # Configuration
    f = open(args.conf_path)
    
    conf_text = f.read()
    f.close()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    conf = ConfigFactory.parse_string(conf_text)

    file_backup(os.path.join(args.savedir, 'recording'), args.conf_path, dir_lis=conf['general']['base_exp_dir'])
    
    dist.init()
    rank = dist.get_local_rank()
    global_rank = dist.get_global_rank()
    device = torch.device(rank)
    world_size = dist.get_world_size()
    print(f'Starting in machine {device} which is at rank {global_rank} of world size {world_size} and rank {rank}')
    dist.print0(f'\n\nDistributing across {world_size} GPUs\n\n')
       
    training = ScanTrainer(conf, world_size, rank, device, global_rank, ckpt_path=args.ckpt_path, savedir=args.savedir,    unfreeze_time_for_pca=args.unfreeze_time_for_pca, ngpus=args.ngpus, num_workers=args.num_workers, accumulate_gradients=args.accumulate_gradients, dataset=dataset, opt=opt, opt_hair=opt_hair, pipe=pipe,  pointcloud_path_head=pointcloud_path_head, ip=ip, port=port, prefix2=args.prefix2, scene=args.scene, upsample_hairstyle=args.upsample_hairstyle,  upsample_resolution=args.upsample_resolution, optimize_appearance=args.optimize_appearance, num_steps_coarse=args.num_steps_coarse)         
    training.train(world_size,  rank, device, global_rank)

    #dist.cleanup()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--ckpt_path', default='', type=str)
    parser.add_argument('--savedir', default='./experiments', type=str)
    parser.add_argument('--conf_path', default='./configs/base.conf', type=str)
    parser.add_argument('--ngpus', default=-1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--accumulate_gradients', default=1, type=int)
    parser.add_argument('--unfreeze_time_for_pca', default=-1, type=int)
    parser.add_argument('--upsample_hairstyle', default=False, type=bool)
    parser.add_argument("--prefix2", type=str, default = '')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.10")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--pointcloud_path_head", type=str, default = None)
    parser.add_argument("--hair_conf_path", type=str, default = None)
    parser.add_argument("--scene", type=str, default = '')
    
    parser.add_argument('--upsample_resolution', type=int, default=64)
    parser.add_argument('--num_steps_coarse', type=int, default=20)
    parser.add_argument('--optimize_appearance', type=bool, default=False)
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    
        # Initialize system state (RNG)
    safe_state(args.quiet)

        # Configuration of hair strands
    with open(args.hair_conf_path, 'r') as f:
        replaced_conf = str(yaml.load(f, Loader=yaml.Loader)).replace('DATASET_TYPE', 'monocular')
        opt_hair = yaml.load(replaced_conf, Loader=yaml.Loader)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    

    main(args, lp.extract(args), op.extract(args), opt_hair, pp.extract(args), args.pointcloud_path_head, args.ip, args.port)
 
