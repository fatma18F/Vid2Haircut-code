import trimesh
import torch


def save_strands(global_strands, save_name, num_points=100, cols=None):
    if cols is None:
        cols = torch.cat((torch.rand(global_strands.shape[0], 3).unsqueeze(1).repeat(1, num_points, 1), torch.ones(global_strands.shape[0], num_points, 1)), dim=-1).reshape(-1, 4).cpu() 
    _ = trimesh.PointCloud(global_strands.reshape(-1,3).detach().cpu().numpy(), colors=cols).export(save_name)  