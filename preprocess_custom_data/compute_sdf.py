import trimesh

import os
import torch
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.io import load_ply
import numpy as np
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.structures import Pointclouds
from plyfile import PlyData

import trimesh
import numpy as np
import trimesh.proximity
import torch.nn.functional as F

def mesh_to_sdf(path, grid_size=64):
    # Load the mesh
    mesh = trimesh.load(path, process=True)

    bbox = mesh.bounds
    min_bound = bbox[0] - 0.02
    max_bound = bbox[1] + 0.02
    lin = [np.linspace(min_bound[i], max_bound[i], grid_size) for i in range(3)]
    grid = np.stack(np.meshgrid(*lin, indexing='ij'), axis=-1)
    points = grid.reshape(-1, 3)

    # Fast signed distance directly
    sdf = trimesh.proximity.signed_distance(mesh, points)

    # Reshape to grid
    sdf_grid = sdf.reshape((grid_size, grid_size, grid_size))
    sdf_grid *= -1
    return sdf_grid, points, min_bound, max_bound


def make_sdf_tensor(sdf_grid_np, min_bound, max_bound, device='cpu'):
    """
    Convert a NumPy sdf_grid of shape (D,H,W) into a torch tensor
    of shape (1,1,D,H,W), moved to `device`.
    """
    sdf = torch.from_numpy(sdf_grid_np).float().to(device)
    # unsqueeze to (B=1, C=1, D, H, W)
    return sdf.unsqueeze(0).unsqueeze(0), \
           torch.tensor(min_bound, dtype=torch.float32, device=device), \
           torch.tensor(max_bound, dtype=torch.float32, device=device)



def query_sdf(sdf_tensor, bounds_min, bounds_max, points):
    """
    sdf_tensor : (1,1,D,H,W)
    bounds_min : (3,) tensor
    bounds_max : (3,) tensor
    points     : (N,3) tensor of xyz coordinates in world space
    
    returns: sdf_vals (N,) differentiable w.r.t. points (and sdf_tensor if desired)
    """
    B, C, D, H, W = sdf_tensor.shape
    
    # normalize points to [-1,1]
    p_norm = 2.0 * (points - bounds_min[None]) / (bounds_max - bounds_min)[None] - 1.0
    # reshape to (B, N, 1, 1, 3)
    grid = p_norm.view(1, -1, 1, 1, 3)
    
    grid = grid[..., [2,1,0]]

    # trilinear sample
    sampled = F.grid_sample(
        sdf_tensor,      # (1,1,D,H,W)
        grid,            # (1,N,1,1,3)
        mode='bilinear',
        align_corners=True
    )
    # sampled is (1,1,N,1,1) → flatten to (N,)
    return sampled.view(-1)



def process_sequence(seq_name, root_path, grid_size=64, start=0, end=-1):
    #input_path = os.path.join(root_path, seq_name)
    sdf_path = os.path.join(root_path, f"{seq_name}_sdfs")
    min_path = os.path.join(root_path, f"{seq_name}_mins")
    max_path = os.path.join(root_path, f"{seq_name}_maxs")

    #os.makedirs(sdf_path, exist_ok=True)
    #os.makedirs(min_path, exist_ok=True)
    #os.makedirs(max_path, exist_ok=True)

    # mesh_files = sorted(os.listdir(input_path))
    # if end == -1:
    #     end = len(mesh_files)
        
    
    mesh_files=["/home/ayed/prerocessing_input_monocular_hair_modelling/sim_data/data/flame/head_0000.obj"]
    input_path="/home/ayed/prerocessing_input_monocular_hair_modelling/sim_data/data/flame/"
    for mesh_file in mesh_files:#[start:end]:
        name = "sim" #os.path.splitext(mesh_file)[0]
        print("name",name)
        sdf_file = os.path.join(root_path, f"sdf_grid_{name}.npy")
        min_file = os.path.join(root_path, f"min_bound_{name}.npy")
        max_file = os.path.join(root_path, f"max_bound_{name}.npy")

        # Skip if all output files already exist
#         if os.path.exists(sdf_file) and os.path.exists(min_file) and os.path.exists(max_file):
#             print(f"Skipping {mesh_file}, output already exists.")
#             continue

        print(mesh_file)
        mesh_full_path = os.path.join(input_path, mesh_file)
        sdf_grid, points, min_bound, max_bound = mesh_to_sdf(mesh_full_path, grid_size=grid_size)

        np.save(sdf_file, sdf_grid)
        np.save(min_file, min_bound)
        np.save(max_file, max_bound)
        
        

# Main logic
id = '140'
root_path = f"/home/ayed/prerocessing_input_monocular_hair_modelling/person{id}/"
root_path='/home/ayed/prerocessing_input_monocular_hair_modelling/synth_data/05-1_straight-hair-M_2/'


root_path="/home/ayed/prerocessing_input_monocular_hair_modelling/sim_data/data/"
sequences = [ 'smpl_mesh']
for seq in sequences:
    process_sequence(seq, root_path, grid_size=48, start=0, end=-1)
