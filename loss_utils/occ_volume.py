import trimesh
import torch
import numpy as np
import torch
import torch
import torch.nn.functional as F

class VolumePrior:
    def __init__(self,
                 min_bound=[-0.32057703,  0.929, -0.28714779],
                 voxel_size=0.002,
                 resolution= [320, 548, 302],
                 device='cuda',
                 tol=1e-8
                ):
        
        self.device = device
        
        self.min_bound = torch.tensor(min_bound, device=self.device, dtype=torch.float)
        self.max_bound = (self.min_bound + voxel_size * torch.tensor(resolution, device=self.device))
        
        self.voxel_size = voxel_size
        
        self.resolution = resolution  
        self.mgrid = self._setup_mgrid() 
    
  
    def query_pts_occ_grid_sampling(self, pc, occ_field):
        '''
        occ field: [N, D, H, W] where D corresponds to depth (z-axis)
        pc: [N, M, 3] where 3 corresponds to (x, y, z)

        -----------------------
        Return: occupancy at each point. 
        '''

        # Permute occ_field to ensure depth (z) is the first dimension
        occupancy_volume_expanded = occ_field.permute(0, 3, 2, 1).float().unsqueeze(1)  # Changes shape to [N, H, W, D]

        normalized_grid = self.world2local(pc)

        N = pc.shape[0]
        M = pc.shape[1]

        grid = normalized_grid.view(N, M, 1, 1, 3)

        output = F.grid_sample(occupancy_volume_expanded, grid, mode='bilinear', align_corners=True)
        
        vals = torch.where(output.reshape(-1) > 0)[0]
        occ_rate = vals.shape[0] / pc.reshape(-1, 3).shape[0]
        
        return output.reshape(N, M), occ_rate
    
    
    def query_pts_dir_grid_sampling(self, pc, dir_field):
        '''
        occ field: [N, D, H, W] where D corresponds to depth (z-axis)
        pc: [N, M, 3] where 3 corresponds to (x, y, z)

        -----------------------
        Return: occupancy at each point. 
        '''

        # Permute occ_field to ensure depth (z) is the first dimension
        dir_volume_expanded = dir_field.permute(0, 1, 4, 3, 2).float()  # Changes shape to [N, H, W, D]

        normalized_grid = self.world2local(pc)

        N = pc.shape[0]
        M = pc.shape[1]
        
        grid = normalized_grid.view(N, M, 1, 1, 3)

        output = F.grid_sample(dir_volume_expanded, grid, mode='bilinear', align_corners=True).permute(0, 2, 1, 3, 4)

        return output.reshape(N, M, 3)#, occ_rate
    
    

    def world2local(self, pc):
        
        '''
        Input:
        pc [M, N, 3]
        ------------------------------------
        Return: normalized pc in range [-1, 1]
        '''

        normalized_pc = 2 * (pc - self.min_bound[None][None]) / (self.max_bound[None][None] - self.min_bound[None][None]) - 1
        return normalized_pc
    
    
    def _setup_mgrid(self):
        
        start_x, end_x, step_x = self.min_bound[0] + self.voxel_size / 2, self.min_bound[0] + self.voxel_size * self.resolution[0], self.voxel_size
        start_y, end_y, step_y = self.min_bound[1] + self.voxel_size / 2, self.min_bound[1] + self.voxel_size * self.resolution[1], self.voxel_size
        start_z, end_z, step_z = self.min_bound[2] + self.voxel_size / 2, self.min_bound[2] + self.voxel_size * self.resolution[2], self.voxel_size

        x_range = torch.arange(start_x, end_x, step_x).to(self.device)
        y_range = torch.arange(start_y, end_y, step_y).to(self.device)
        z_range = torch.arange(start_z, end_z, step_z).to(self.device)

        # Create a 3D grid
        x_grid, y_grid, z_grid = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
        grid_points = torch.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], dim=-1)

        return grid_points
    
    
    def save_mesh(self, mgrid, occ_field, save_path='debug.ply'):
        
#         mgrid = self.mgrid.reshape(occ_field.shape[0], -1, 3)
        
#         print('mgrid', mgrid.max(), mgrid.min())
        
        occ, _ = self.query_pts_occ_grid_sampling(mgrid, occ_field)
        
#         print('mgrid', mgrid.shape, occ_field.shape, occ.shape)
#         occ = self.query_point_occ(mgrid, occ_field)

        filter_idx = (occ > 0).float()[..., None]
        occ_indexes = torch.where(filter_idx > 0)[0]
        
#         print(occ_indexes.shape, mgrid.shape, mgrid[0, occ_indexes].shape)
        occupied = mgrid[0, occ_indexes]
#         print('occ', occupied.shape)
        _ = trimesh.PointCloud(occupied.reshape(-1, 3).detach().cpu().numpy()).export(save_path)
    

    def points_to_voxel(self, points):
                
        indexs = (points - self.min_bound) / self.voxel_size
        
        voxels_idxs = indexs.floor().long()
        
        
        weight = indexs - indexs.floor().long()

        clipped_voxels = voxels_idxs.clip(torch.zeros(3, device=self.device).long(),
                                          torch.tensor(self.resolution, device=self.device).long()-1
                                         )
        
        return clipped_voxels, weight.float()
    
    
    def query_point_dir(self, point, dir_field):
            
            ''' non differentiable version'''
            # point of size [bs, N, 3]
            # occ_field of size [bs,F, H, W, D]
            
            bs = point.shape[0]
            N_points = point.shape[1]
            
            idxs, _ = self.points_to_voxel(point)
                        
            batch_indices = torch.arange(bs).unsqueeze(1).expand(bs, N_points)
            x, y, z = idxs[:, :, 0],  idxs[:, :, 1],  idxs[:, :, 2]
            
            return dir_field[batch_indices, :, x, y, z]
        

    def query_point_occ(self, point, occ_field):
            
            ''' non differentiable version'''
            # point of size [bs, N, 3]
            # occ_field of size [bs, H, W, D]
            
            bs = point.shape[0]
            N_points = point.shape[1]
            
            idxs, _ = self.points_to_voxel(point)
                        
            batch_indices = torch.arange(bs).unsqueeze(1).expand(bs, N_points)
            x, y, z = idxs[:, :, 0],  idxs[:, :, 1],  idxs[:, :, 2]

            return occ_field[batch_indices, x, y, z]