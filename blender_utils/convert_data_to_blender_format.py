import os
import argparse
import numpy as np
import trimesh
from plyfile import PlyData
import pickle

NUM_STRANDS = 50000
L = 200

def load_transform(scale_path):
    with open(scale_path, 'rb') as f:
        transform = pickle.load(f)
    scale_mat = np.eye(4, dtype=np.float32)
    scale_mat[:3, :3] *= transform['scale']
    scale_mat[:3, 3] = np.array(transform['translation'])
    return transform, scale_mat

def save_selected_cameras(cam_path, scale_mat, save_dir, indices=[0, 1]):
    camera_dict = np.load(cam_path)
    proj_matrix = []
    for world_mat in camera_dict:
        P = world_mat @ scale_mat
        proj_matrix.append(P[:3, :4])
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'selected_cam.npy'), np.stack(proj_matrix)[indices])

def process_and_save_mesh(head_path, transform, save_path):
    mesh = trimesh.load(head_path, process=False)
    points = np.array(mesh.vertices)
    points = (points - transform['translation']) / transform['scale']
    transformed_mesh = trimesh.Trimesh(vertices=points, faces=mesh.faces)
    transformed_mesh.export(save_path)

def process_and_save_strands(ply_path, transform, save_path, scene_name):
    strands_ply = PlyData.read(ply_path).elements[0].data
    raw_points = np.stack([strands_ply['x'], strands_ply['y'], strands_ply['z']], axis=1)
    points = raw_points
    points = (points - transform['translation']) / transform['scale']
    points = np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=1)
    strands_npy = points.reshape(-1, L, 3)
    np.save(os.path.join(save_path, scene_name), strands_npy)

def main(args):
    transform, scale_mat = load_transform(args.scale_path)
    save_selected_cameras(args.cam_path, scale_mat, args.save_dir)

    frame_id = args.frame_id
    scene_path = os.path.join(args.scene_path, args.exp_name, f'{frame_id}/pointclouds_train')
    
    scene_names = sorted([
        name for name in os.listdir(scene_path)
        if 'pred_0' in name
    ])

    save_strands_path = os.path.join(args.save_dir, args.exp_name)
    save_head_path = os.path.join(args.save_dir, 'bodys')
    os.makedirs(save_head_path, exist_ok=True)
    os.makedirs(save_strands_path, exist_ok=True)

    tracker_files = sorted(os.listdir(args.tracker_path))
    print(save_strands_path)
    for idx, scene_name in enumerate(scene_names):
        print(scene_name)
        head_name = tracker_files[0]
        head_path = os.path.join(args.tracker_path, head_name)
        process_and_save_mesh(head_path, transform, os.path.join(save_head_path, head_name))

        path_to_strands = os.path.join(scene_path, scene_name)
        process_and_save_strands(path_to_strands, transform, save_strands_path, head_name.split('.')[0])

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transform and save strands and head mesh.")
    parser.add_argument('--exp_name', type=str, default='static.conf', help='Experiment name')
    parser.add_argument('--scene_path', type=str, default='/home/vsklyarova/Projects/monocular-hair-modeling/exps_inverse_stage', help='Path to pointclouds_train')
    parser.add_argument('--frame_id', type=str, default='000001', help='Path to pointclouds_train')
    parser.add_argument('--save_dir', type=str, default='./example_data', help='Where to save outputs')
    parser.add_argument('--data_path', type=str, default='./example_data/000001', help='Dataset base path (for scale/cam)')
    parser.add_argument('--tracker_path', type=str, default='./example_data/smpl_mesh', help='Path to predicted head meshes')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')

    args = parser.parse_args()

    # Derive dependent paths
    args.scale_path = os.path.join(args.data_path, 'scale.pickle')
    args.cam_path = os.path.join(args.data_path, 'cameras.npy')

    main(args)