import numpy as np
import trimesh

import sys
import trimesh
import yaml
from PIL import Image
import os
from tqdm import tqdm
import pickle
import torch
import cv2

import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizer,
    PointsRenderer,
    PointsRasterizationSettings,
    AlphaCompositor,
)
from pytorch3d.structures import Pointclouds

from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
import sys
import pickle
sys.path.append('./submodules/gaussian-splatting-hair/ext/NeuralHaircut')
from NeuS.models.dataset import load_K_Rt_from_P

from hair_rasterizer_opengl import HairRasterizer, PointCloudRasterizer
from visual_utils.camera import postprocess_with_shifted_principal_point

import os
import sys
import torch
import numpy as np
import trimesh
from pathlib import Path
from tqdm import tqdm
from pyhocon import ConfigFactory
from PIL import Image
import cv2
import yaml
import argparse
from skimage.draw import polygon
from pytorch3d.io import (
    load_obj, 
    load_objs_as_meshes, 
    save_obj, 
    load_ply, 
    save_ply
)
from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_scene
from pytorch3d.ops import knn_points
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from pytorch3d.renderer import TexturesVertex, look_at_view_transform

def create_scalp_mask(vis_face, scalp_uvs, resolution=256):
    """
    Creates a binary mask for the scalp area based on visible faces and UVs.

    Args:
        vis_face: Visible face indices.
        scalp_uvs: Scalp UV coordinates.
        resolution: Resolution of the output mask.

    Returns:
        Binary scalp mask as a NumPy array.
    """
    img = np.zeros((resolution, resolution, 1), dtype='uint8')
    
    for i in range(vis_face.shape[0]):
        text = scalp_uvs[vis_face[i]].reshape(-1, 2).cpu().numpy()
        poly = 255 / 2 * (text + 1)
        rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
        img[rr, cc, :] = 255

    return img.transpose(1, 0, 2)

def create_visibility_map(camera, rasterizer, mesh, hair_mask):
    """
    Creates visibility maps for vertices and faces of a mesh given a camera view.

    Args:
        camera: PyTorch3D camera object.
        rasterizer: PyTorch3D rasterizer object.
        mesh: Mesh to compute visibility for.
        hair_mask: Mask to apply for filtering.

    Returns:
        vertex_visibility_map: Visibility map for vertices.
        faces_visibility_map: Visibility map for faces.
        pix_to_face: Mapping of pixels to face indices.
    """
    fragments = rasterizer(mesh, cameras=camera)
    pix_to_face = fragments.pix_to_face * hair_mask[None].unsqueeze(-1)
    packed_faces = mesh.faces_packed()
    packed_verts = mesh.verts_packed()

    vertex_visibility_map = torch.zeros(packed_verts.shape[0])
    faces_visibility_map = torch.zeros(packed_faces.shape[0])
    #if len(pix_to_face.unique())==2:
    #    visible_faces = pix_to_face.unique()[1:]  # Exclude -1
    #else:
    visible_faces = pix_to_face.unique()[2:]  # Exclude -1
    visible_verts_idx = packed_faces[visible_faces]
    unique_visible_verts_idx = torch.unique(visible_verts_idx)

    vertex_visibility_map[unique_visible_verts_idx] = 1.0
    faces_visibility_map[torch.unique(visible_faces)] = 1.0

    return vertex_visibility_map, faces_visibility_map, pix_to_face

def check_visiblity_of_faces(cams, meshRasterizer, full_mesh, mesh_head, n_views=2, hair_mask=''):
    """
    Checks the visibility of mesh faces from multiple camera views.

    Args:
        cams: List of camera objects.
        meshRasterizer: PyTorch3D rasterizer.
        full_mesh: Full mesh object.
        mesh_head: Mesh of the head. head_path='./data/head_prior.obj'
        n_views: Minimum number of views for visibility.
        hair_mask: Hair mask to apply.

    Returns:
        vertex: Visible vertices of the mesh.
        face_faces: Visible faces of the mesh.
        pix_to_face: Pixel-to-face mapping.
        face_idx: Indices of visible faces relative to the head mesh.
    """
    vis_maps = []
    for cam in range(len(cams)):
        v, _, pix_to_face = create_visibility_map(cams[cam], meshRasterizer, full_mesh, hair_mask=hair_mask)
        vis_maps.append(v)

    vis_mask = (torch.stack(vis_maps).sum(0) > n_views).float()

    idx = torch.nonzero(vis_mask).squeeze(-1).tolist()
    idx = [i for i in idx if i > mesh_head.verts_packed().shape[0]]

    full_scalp_idx = sorted(idx)
    indices_mapping = {j: i for i, j in enumerate(idx)}

    face_faces = []
    face_idx = torch.tensor(idx).to('cuda')
    
#     print('near 154',face_idx)
    vertex = full_mesh.verts_packed()[face_idx]

    good_idx = []
    for fle, i in enumerate(full_mesh.faces_packed()):
        if i[0] in face_idx and i[1] in face_idx and i[2] in face_idx:
            face_faces.append([
                indices_mapping[i[0].item()],
                indices_mapping[i[1].item()],
                indices_mapping[i[2].item()]
            ])
            good_idx.append(fle)

    return vertex, torch.tensor(face_faces), pix_to_face, face_idx - mesh_head.verts_packed().shape[0]








import torch.nn as nn
class ScalpRenderer(nn.Module):
    def __init__(
        self,
        #head_path='./inputs/data/head_prior.obj', '/home/ayed/data/nersemble-data/notebooks/head_prior_VHAPAligned2.obj'
        #scalp_path='./inputs/data/scalp_all_data.obj', '/home/ayed/data/nersemble-data/notebooks/scalp_all_data_VHAPAligned.obj' 
        head_path='./inputs/data/head_prior.obj', 
        scalp_path='./inputs/data/scalp_all_data.obj',
        uv_path = "./inputs/data/uvs_full_blender.pth",
        device='cuda',
        size=(512, 512),
        gs_scale_path='',
    ):
        
        super(ScalpRenderer, self).__init__()
        
        self.mesh_head = load_objs_as_meshes([head_path], device=device)
        self.mesh_hair = load_objs_as_meshes([scalp_path], device=device)
        self.device = device
        self.gs_scale_path=gs_scale_path
       

        self.uvs = torch.load(uv_path)
        translate_to_sphere = torch.tensor([-0.0037,  1.6835,  0.0071], device=device) 

        scale_to_sphere = torch.tensor(0.2176, device=device)


        self.mesh_head.offset_verts_(-translate_to_sphere)
        self.mesh_head.scale_verts_((1.0 / float(scale_to_sphere)))

        self.mesh_hair.offset_verts_(-translate_to_sphere)
        self.mesh_hair.scale_verts_((1.0 / float(scale_to_sphere)))

 
        
        self.size = size

        self.raster_settings_mesh = RasterizationSettings(
                            image_size=self.size, 
                            blur_radius=0.001, 
                            faces_per_pixel=2, 
                        )


                # init camera
        R = torch.tensor([[[ 0.9985,  0.0528,  0.0107],
                 [ 0.0521, -0.9971,  0.0560],
                 [ 0.0136, -0.0553, -0.9984]]], device=device)

        t =  torch.tensor([[-0.3067, -0.0532,  6.6122]], device=device)


        cam_intr = torch.tensor([[[855.0901,  28.6012, 298.8687,   0.0000],
                 [  0.0000, 843.7341, 215.4442,   0.0000],
                 [  0.0000,   0.0000,   1.0000,   0.0000],
                 [  0.0000,   0.0000,   0.0000,   1.0000]]], device=device)

        cam = cameras_from_opencv_projection(
                                            camera_matrix=cam_intr.cuda(), 
                                            R=R.cuda(),
                                            tvec=t.cuda(),
                                            image_size=torch.tensor(self.size)[None].cuda()
                                              ).cuda()

        # init mesh rasterization
        self.meshRasterizer = MeshRasterizer(cam, self.raster_settings_mesh)
        
        #Hair/scalp verts are white (1,1,1); head/bust verts are black (0,0,0).
        self.mesh_hair.textures = TexturesVertex(verts_features=torch.ones_like(self.mesh_hair.verts_packed()).float().cuda()[None])
        self.mesh_head.textures = TexturesVertex(verts_features=torch.zeros_like(self.mesh_head.verts_packed()).float().cuda()[None])

        # join hair and bust mesh to handle occlusions
        self.full_mesh = join_meshes_as_scene([self.mesh_head, self.mesh_hair])

    
    def upload_cam(self, cam):
        intrinsics, pose = load_K_Rt_from_P(None, cam.detach().cpu().numpy())
        intrinsics_all = torch.tensor(intrinsics).to(self.device).float()  
        pose_all_inv = torch.inverse(torch.tensor(pose)).to(self.device).float()

        #undo scale
        scale_mat = np.eye(4, dtype=np.float32)
        with open(self.gs_scale_path, 'rb') as f:
            transform = pickle.load(f)
            scale_mat[:3, :3] *= transform['scale']
            scale_mat[:3, 3] = transform['translation']

            scale_inv = np.eye(4, dtype=np.float32)
            scale_inv[:3, :3] = np.linalg.inv(scale_mat[:3, :3])  # Inverse of scaling matrix
            scale_inv[:3, 3] = -scale_mat[:3, 3] / scale_mat[:3, :3].diagonal()  # Inverse translation
            pose_all_inv = pose_all_inv @ torch.tensor(scale_inv).cuda()



        cams = cameras_from_opencv_projection(
                                        camera_matrix=intrinsics_all[None].to(self.device), 
                                        R=pose_all_inv[:3, :3][None].to(self.device),
                                        tvec=pose_all_inv[:3, 3][None].to(self.device),
                                        image_size=torch.tensor(self.size)[None].to(self.device)
                                         ).to(self.device)
        
        

        return cams
    
        
    def forward(self, cam, hair_mask):
        
        cam = self.upload_cam(cam)
#         print(cam, self.full_mesh)
        #self.meshRasterizer = MeshRasterizer(cam, self.raster_settings_mesh)
        
        vis_vertex, vis_face, pix_to_face, full_list_idx = check_visiblity_of_faces([cam], self.meshRasterizer, self.full_mesh, self.mesh_head, n_views=0, hair_mask=hair_mask)
        scalp_mask = create_scalp_mask(vis_face, self.uvs[full_list_idx.cpu()].cuda(), resolution=256).squeeze(-1)
        
        return torch.tensor(scalp_mask, device=self.device) / 255.



def project_points(cam_gaus, translate_to_sphere, scale_to_sphere, texture, device='cuda', method='opengl'):
        
    path_to_coords_for_each_origin = os.path.join('./inputs/data/coords_for_each_origin_64x64.pth')
    coords_for_each_origin = torch.load(path_to_coords_for_each_origin).float().to(device).reshape(-1, 3)
                    
    points = (coords_for_each_origin - translate_to_sphere ) / scale_to_sphere  
    
    intrinsics, pose = load_K_Rt_from_P(None, cam_gaus.detach().cpu().numpy())
    intrinsics_all = torch.tensor(intrinsics).to(device).float()  
    pose_all_inv = torch.inverse(torch.tensor(pose)).to(device).float()
    
        
#     check occlusion
    vis_idxes = torch.tensor(occlusion_mask(intrinsics_all, pose_all_inv, translate_to_sphere, scale_to_sphere)).to(device)
    
    point_cloud = Pointclouds(points=[points[vis_idxes]], features=[texture.reshape(-1, 1)[vis_idxes].repeat(1, 3)])
    
    
    cams = cameras_from_opencv_projection(
                                    camera_matrix=intrinsics_all[None].to(device), 
                                    R=pose_all_inv[:3, :3][None].to(device),
                                    tvec=pose_all_inv[:3, 3][None].to(device),
                                    image_size=torch.tensor([512, 512])[None].to(device)
                                     ).to(device)
    
    raster_settings = PointsRasterizationSettings(
                                image_size=512,  # Image resolution
                                radius=0.008,     # Point size
                                points_per_pixel=1,  # Max points to blend per pixel
                                )

    # Initialize rasterizer and compositor
    rasterizer = PointsRasterizer(cameras=cams, raster_settings=raster_settings)
    compositor = AlphaCompositor()

    # Initialize the renderer with the rasterizer and compositor
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

    # Render the point cloud
    rendered_image = renderer(point_cloud, cameras=cams)
    
    

    return rendered_image, rasterizer(point_cloud)[0]

def occlusion_mask(intrinsics, extrinsics, translate_to_sphere, scale_to_sphere, head_prior_path = '.data/head_prior.obj', RESOLUTION=512):

    path_to_coords_for_each_origin = f'./inputs/data/coords_for_each_origin_64x64.pth'
    points = torch.load(path_to_coords_for_each_origin).float().reshape(-1, 3)
#     [torch.where(points.sum(1)>-300)]
    points = points.detach().cpu().numpy()
    points = (points - translate_to_sphere.detach().cpu().numpy()) / scale_to_sphere.detach().cpu().numpy()

    head = trimesh.load(head_prior_path)
#     print(head, points.shape)
    n_points = 2
    H, W = int(RESOLUTION), int(RESOLUTION)
    mesh_pts = (np.array(head.vertices) - translate_to_sphere.detach().cpu().numpy()) / scale_to_sphere.detach().cpu().numpy()
    
    raster = HairRasterizer(points.shape[0], n_points, (mesh_pts, np.array(head.faces)), resolution=(H, W), line_width=0.01)
    
    res = raster.rasterize(
                        torch.tensor(points).cpu(),
                        (intrinsics.cpu(), extrinsics.cpu()),
                        a = torch.randn(torch.tensor(points).shape[0], 1),
                        return_idx=True
                      )[1][0]
    
    valid_pixels = torch.tensor(res[res != -1], device='cpu')
    

    return (valid_pixels // 1).long()
