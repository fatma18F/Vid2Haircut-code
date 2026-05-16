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

from mv_scanner_3d import ScanTrainer
import trimesh
from PIL import Image

             

class VideoScanTrainer(ScanTrainer):
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
        IMG_W=np.array(Image.open(f'{data_root}/{prefix2}/images/{scene}')).shape[0]
        self.use_Vanessa_data=IMG_W==1200
        

        self._init_basic_config(num_steps_coarse, device, ngpus, accumulate_gradients,
                            upsample_hairstyle, upsample_resolution, optimize_appearance, config, 
                            unfreeze_time_for_pca)
        
        self._init_dataset(config, scene, prefix2, world_size, rank, num_workers)

        self._init_config(config)
            
        self._init_roots_and_blend_shapes()

        self._init_gaussian_trainer(dataset, opt, opt_hair, pipe, pointcloud_path_head, ip, port, rank, config)

        self._init_canonical_to_world(config)

        self._setup_dirs_and_writer(savedir)
        
#         setup hairstyle
        self.static_hairstyle_path = static_hairstyle_path
        self.setup_hairstyle(rank, device)
        
#         setup optimizer
        self.optimizer = self.configure_optimizers()
        
#         upload prior sdf
        self._upload_sdf_prior(config)

    
    def training_step(self, batch, batch_idx, world_size, rank, device, global_rank, mode='train'):
        
        if self.step % self.frame_step == 0:
            self.update_iteration(frame_idx=self.step // self.frame_step)
                
        pred_points_vis = self.retrieve()
        
        loss, logs = self.single_step(pred_points_vis, batch, batch_idx, world_size, rank, device, global_rank)
        
        return loss, logs   
    
    
    
    def train(self, world_size, rank, device, global_rank):
        print('train here', self.step, self.frame_name)

        all_frame_num = self.num_frames * self.frame_step 
        all_frame_list = sorted(os.listdir(self.data_path))
        
        print(all_frame_list, self.step // self.frame_step)
        
        
        try:
         with tqdm(total=self.all_steps, desc="Training steps", dynamic_ncols=True, smoothing=0.3) as pbar:

            while self.step < all_frame_num:
            
                #(1) Re-init dataset every `frame_step` steps
                if self.step % self.frame_step == 0:
                    curr_frame_name = all_frame_list[self.step // self.frame_step]
                    print('Updating dataset at step', self.step, curr_frame_name)
                    self._init_dataset(self.config, self.scene, curr_frame_name, world_size, rank, self.num_workers)

                # (2) Train for N steps on the current dataset
                inner_steps = 0
                for batch in self.train_dl:

                    if inner_steps >= self.frame_step:
                        break

                    loss, logs = self.training_step(batch, self.step, world_size, rank, device, global_rank)

                    if self.accumulate_gradients > 1:
                        loss /= self.accumulate_gradients

                    loss.backward()

                    if (self.step + 1) % self.accumulate_gradients == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if self.step % self.logging_freq == 0 and rank == 0:
                        print('Logging at step', self.step)
                        for key in logs:
                            self.writer.add_scalar(f'{key}', logs[key], self.step)

                    self.step += 1
                    inner_steps += 1
                    pbar.update(1)
                    pbar.set_postfix({'loss': float(loss.item())})       
                  

                self.epoch += 1

                if self.step >= self.all_steps:
                    print(f"Training completed at step {self.step}")
                    break

        except KeyboardInterrupt:
            pass
         
        
    

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
    
         
       
    training = VideoScanTrainer(conf, world_size, rank, device, global_rank, ckpt_path=args.ckpt_path, savedir=args.savedir,    unfreeze_time_for_pca=args.unfreeze_time_for_pca, ngpus=args.ngpus, num_workers=args.num_workers, accumulate_gradients=args.accumulate_gradients,dataset=dataset, opt=opt, opt_hair=opt_hair, pipe=pipe,  pointcloud_path_head=pointcloud_path_head, ip=ip, port=port, prefix2=args.prefix2, scene=args.scene, upsample_hairstyle=args.upsample_hairstyle,  upsample_resolution=args.upsample_resolution, optimize_appearance=args.optimize_appearance, num_steps_coarse=args.num_steps_coarse, static_hairstyle_path=args.static_hairstyle_path)

    
    print('i am here in scantrainer')            
                
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
 
