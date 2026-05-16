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

# Dataset modules
# from datasets.dataset_strands_pca_with_bald_map_accelerate_multiview import HairstyleDataset, decode_pca

from src.utils.geometry import decode_pca
from datasets.datasets_scanner import HairstyleRealDatasetScanner

# Losses and model utils
#from losses import temporal_regularization_loss 
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
                 optimize_appearance=False,
                 static_hairstyle_path=''
                 ):
        
        nn.Module.__init__(self)
        data_root=config['dataset'].get('data_path', '')
        use_SingleView=config['dataset_real']['use_SingleView']
        if use_SingleView:
          IMG_W=np.array(Image.open(f'{data_root}/SingleView/{prefix2}/images/{scene}')).shape[0]
        else:
            IMG_W=np.array(Image.open(f'{data_root}/{prefix2}/images/{scene}')).shape[0]

        self.use_Vanessa_data=IMG_W==1200

        self._init_basic_config(num_steps_coarse, device, ngpus, accumulate_gradients,
                            upsample_hairstyle, upsample_resolution, optimize_appearance, config, 
                            unfreeze_time_for_pca)
        print('i am in 3d', num_workers)
        self._init_dataset(config, scene, prefix2, world_size, rank, num_workers)

        self._init_config(config)
            
        self._init_roots_and_blend_shapes()

        self._init_gaussian_trainer(dataset, opt, opt_hair, pipe, pointcloud_path_head, ip, port, rank, config)

        self._setup_dirs_and_writer(savedir)

        self._init_canonical_to_world(config)

        
#         setup hairstyle
        self.static_hairstyle_path = static_hairstyle_path
        self.setup_hairstyle(rank, device)
        self._init_deformation(config, device, rank)

#         setup optimizer
        self.optimizer = self.configure_optimizers()
        
#         upload prior sdf
        self._upload_sdf_prior(config)

    
    def training_step(self, batch, batch_idx, world_size, rank, device, global_rank, mode='train'):
                
        pred_points_vis = self.retrieve()
        self.scanner_type='scanner_3d'
        loss, logs = self.single_step(pred_points_vis, batch, batch_idx, world_size, rank, device, global_rank)
        

        return loss, logs

    
    def setup_hairstyle(self, rank, device):
        self.appearance = None
        selected_strands = torch.tensor(np.array(trimesh.load(self.static_hairstyle_path).vertices)).reshape(-1, self.num_points, 3).float().to(device).to(rank)
        self.origins_up = selected_strands[:, :1]
        local_space = selected_strands - selected_strands[:, :1]
        dirs = (local_space[:, 1:] - local_space[:, :-1])
        processed_dirs = dirs.detach().view(-1, 3).contiguous().clone()
        self.hairstyle_dirs = nn.Parameter(processed_dirs, requires_grad=True)
        

        

    def retrieve(self):
        pts_ = self.origins_up + torch.cat([torch.zeros_like(self.origins_up), torch.cumsum(self.hairstyle_dirs.reshape(-1, 199, 3), dim=1)], dim=1)
        return pts_[None]
    
    
    
    @torch.no_grad()
    def save_model(self):
        pass
    
   
        
    def configure_optimizers(self):


        l = [
            {'params': [self.hairstyle_dirs], 'lr': self.learning_rate/10}
             ]
        
        if self.gaus_trainer.optimize_scale:
            z = {'params': [self.gaus_trainer.scale_head], 'lr': self.learning_rate}
            l.append(z)
            
        if self.optimize_appearance:
            z = {'params': [self.appearance], 'lr': self.learning_rate*500}
            l.append(z)
            print('optimize appearance')

        if self.use_deformation_mlp:
            l.append({
            'params': self.deformerMLP.parameters(),
            'lr': self.learning_rate * 5  # for example
            })
            l.append({
                'params': self.frame_codes.parameters(),
                'lr': self.learning_rate * 5  # same or different LR
            })
            
        if self.optimizer_type == 'adam':
                opt_ae = torch.optim.Adam(l,
                                    lr=self.learning_rate, betas=(0.5, 0.9))
            
        elif self.optimizer_type == 'adamw':
            print('create adamw', self.learning_rate, self.weight_decay)
            
            opt_ae = torch.optim.AdamW(l,
                                  lr=self.learning_rate, weight_decay=self.weight_decay)
            
            
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
    
         
       
    training = ScanTrainer(conf, world_size, rank, device, global_rank, ckpt_path=args.ckpt_path, savedir=args.savedir,    unfreeze_time_for_pca=args.unfreeze_time_for_pca, ngpus=args.ngpus, num_workers=args.num_workers, accumulate_gradients=args.accumulate_gradients, dataset=dataset, opt=opt, opt_hair=opt_hair, pipe=pipe,  pointcloud_path_head=pointcloud_path_head, ip=ip, port=port, prefix2=args.prefix2, scene=args.scene, upsample_hairstyle=args.upsample_hairstyle,  upsample_resolution=args.upsample_resolution, optimize_appearance=args.optimize_appearance, num_steps_coarse=args.num_steps_coarse, static_hairstyle_path=args.static_hairstyle_path)

    
                
                
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
    parser.add_argument("--static_hairstyle_path", type=str, default = '')
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
    parser.add_argument('--num_steps_coarse', type=int, default=200)
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
 
