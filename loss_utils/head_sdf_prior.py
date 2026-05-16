import numpy as np
import sys
sys.path.append('/home/ayed/NeuSurf/models')
from fields import SDFNetwork
import torch.nn as nn
import torch


class SDFHeadPrior:
    def __init__(self, path_to_ckpt, apply_relu=False, device='cuda', tol=1e-8):
        self.sdf = SDFNetwork(d_in = 3,
                 d_out = 257,
                 d_hidden = 256,
                 n_layers = 8,
                 skip_in=(4,),
                 multires=6,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False)
        
        checkpoint = torch.load(path_to_ckpt, map_location=device)
        self.sdf.load_state_dict(checkpoint['sdf_network_fine'])
        self.sdf.to(device)
        
        self.apply_relu = apply_relu
        self.relu = nn.ReLU()
        
        self.scale_to_neus = torch.tensor([0.217], device=device)
        self.translate_to_neus = torch.tensor([-0.0037,  1.6835,  0.0071], device=device)[None]
        

    def forward(self, pts, scale_pts=False):
        if scale_pts:
            pts -= self.translate_to_neus
            pts /= self.scale_to_neus

       
        if self.apply_relu:
            
            return self.relu(self.sdf.sdf(pts))
        
        else:
            return self.sdf.sdf(pts)
    
  