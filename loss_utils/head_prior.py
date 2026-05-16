import numpy as np
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.ops import  knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d import _C
import torch
import pickle
        
import torch
import torch.nn.functional as F
import numpy as np

from loss_utils.point2mesh_distance import point_mesh_face_distance #one way
# from pytorch3d.loss import point_mesh_face_distance

class HeadPriorMesh:
    def __init__(self, path_to_mesh,  device='cuda', tol=1e-8):
        verts, faces, aux = load_obj(path_to_mesh, device='cuda')
        self.mesh =  Meshes(verts=[(verts).float().to(device)], faces=[faces.verts_idx.to(device)])
            
        self.tol = tol
    
    def get_vertex_normals(self, idx):
        return self.mesh.verts_normals_packed()[idx].squeeze(1)
    
    def get_vertex(self, idx):
        return self.mesh.verts_packed()[idx].squeeze(1)
    
    def get_faces_normals(self, idx):
        return self.mesh.faces_normals_packed()[idx]
    
    def points_on_mesh(self, points):
        # Return idxes of points not on mesh
        dists, idxs, pp = self.points2face(points)
        out_idxes = torch.where(dists > self.tol)[0] 
        on_idxes = torch.where(dists <= self.tol)[0] 
        return out_idxes, on_idxes
    
    def sample_from_mesh(self, num_points=512*128):
        points_mesh, points_normals = sample_points_from_meshes(self.mesh, num_samples=num_points, return_normals=True)
        return points_mesh.squeeze(0), points_normals.squeeze(0) #[num_points, 3]
    
    def points2face(self, points):
        pcl = Pointclouds(points=[points.float()])
        points = pcl.points_packed()
        points_first_idx = pcl.cloud_to_packed_first_idx()
        max_points = pcl.num_points_per_cloud().max().item()
        verts_packed = self.mesh.verts_packed()
        faces_packed = self.mesh.faces_packed()
        tris = verts_packed[faces_packed]
        tris_first_idx = self.mesh.mesh_to_faces_packed_first_idx()
        # Compute point to face distance
        dists, idxs = _C.point_face_dist_forward(points.float(), points_first_idx, tris.float(), tris_first_idx, max_points, 5e-3)
        pp = tris[idxs].mean(1)
        # Return idx of closest face, distance and center point of closest face
        return dists, idxs, pp
    
    def compute_distance_of_points(self, points):
        _, idxs, pp = self.points2face(points)
        cosine = ((points - pp) * self.get_faces_normals(idxs)).sum(dim=1)
        idxs = torch.where(cosine < 0)[0] 
        pcl = Pointclouds(points=[points[idxs].float()])
        f = point_mesh_face_distance(self.mesh, pcl)
        return idxs, point_mesh_face_distance(self.mesh, pcl)
        
    def compute_inside_points(self, points):
        _, idxs, pp = self.points2face(points)
        cosine = ((points - pp) * self.get_faces_normals(idxs)).sum(dim=1)
        return torch.where(cosine < 0)[0] 
        
        

class SDFQuery:
    def __init__(self, sdf_path='./heads/sdf.npy',
                 bounds_min='./heads/min.npy',
                 bounds_max='./heads/max.npy',
                 grid_size=64,
                device='cuda'):
        """
        sdf_tensor : (1,1,D,H,W) tensor representing the SDF grid
        bounds_min : (3,) tensor for minimum XYZ bounds
        bounds_max : (3,) tensor for maximum XYZ bounds
        """
        
        self.device = device
        self.grid_size = grid_size
        self.sdf_tensor = torch.from_numpy(np.load(sdf_path)).float().to(self.device).unsqueeze(0).unsqueeze(0)
        self.bounds_min = torch.tensor(np.load(bounds_min), dtype=torch.float32, device=self.device)
        self.bounds_max = torch.tensor(np.load(bounds_max), dtype=torch.float32, device=self.device)
        self.normals = self.compute_sdf_normals().permute(3, 0, 1, 2).unsqueeze(0) 
        self.feat_tenzor = torch.cat((self.sdf_tensor, self.normals), 1)

    def compute_sdf_normals(self):
        """
        sdf_grid: [D, H, W] torch tensor (3D SDF grid)
        voxel_size: float
        Returns: [D, H, W, 3] tensor of unit normals
        """
        sdf_grid = self.sdf_tensor.squeeze(0).squeeze(0)
        D, H, W = sdf_grid.shape

        # Compute finite differences in central region
        dx = (sdf_grid[2:, 1:-1, 1:-1] - sdf_grid[:-2, 1:-1, 1:-1]) / (2 * self.grid_size)
        dy = (sdf_grid[1:-1, 2:, 1:-1] - sdf_grid[1:-1, :-2, 1:-1]) / (2 * self.grid_size)
        dz = (sdf_grid[1:-1, 1:-1, 2:] - sdf_grid[1:-1, 1:-1, :-2]) / (2 * self.grid_size)

        normals = torch.stack([dx, dy, dz], dim=-1)  # [D-2, H-2, W-2, 3]
        normals = F.normalize(normals, dim=-1)

        # Pad manually: replicate border values
        def replicate_border(tensor):
            # tensor: [D, H, W, C]
            tensor = torch.cat([tensor[0:1], tensor, tensor[-1:]], dim=0)  # depth
            tensor = torch.cat([tensor[:, 0:1], tensor, tensor[:, -1:]], dim=1)  # height
            tensor = torch.cat([tensor[:, :, 0:1], tensor, tensor[:, :, -1:]], dim=2)  # width
            return tensor

        normals_padded = replicate_border(normals)
        return normals_padded  # [D, H, W, 3]



    def query(self, points):
        """
        points : (N,3) tensor of XYZ coordinates in world space

        Returns:
            sdf_vals : (N,) tensor of sampled SDF values
        """
        points = points.clamp(self.bounds_min, self.bounds_max)
        
#         print('inside mano distance func', points.requires_grad)
        
        B, C, D, H, W = self.sdf_tensor.shape

        # Normalize points to [-1, 1]
        p_norm = 2.0 * (points - self.bounds_min[None]) / (self.bounds_max - self.bounds_min)[None] - 1.0
        grid = p_norm.view(1, -1, 1, 1, 3)
        grid = grid[..., [2, 1, 0]]  # Swap to (z, y, x) order expected by grid_sample

        # Trilinear sampling
        sampled = F.grid_sample(
            self.feat_tenzor,
            grid,
            mode='bilinear',
            align_corners=True
        )
        

        # Flatten result
        sdf_vals = sampled[:, :1].view(-1)
        sdf_normals = sampled[:, 1:].permute(0, 2, 1, 3, 4).squeeze(0).squeeze(-1).squeeze(-1)
        
        return sdf_vals, sdf_normals
