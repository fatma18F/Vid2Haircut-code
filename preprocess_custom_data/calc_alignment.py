import cv2 as cv
import os 
from torchvision import transforms
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import argparse
from PIL import Image
import os
import face_alignment   
import cv2
import sys
import imageio
sys.path.append(os.path.join(sys.path[0], './utils'))
from skimage.transform import resize
import torch.nn.functional as F

def load_apple_pro_depth_and_normalize(path, hair_mask):
    depth = np.load(path.replace('jpg', 'npz'))['depth']

    hair_silh = hair_mask > 0.5
    depth_inside_silhouette = depth[hair_silh]
    # Compute min and max within the silhouette
    min_depth = np.min(depth_inside_silhouette)
    max_depth = np.max(depth_inside_silhouette)

    # Normalize depth within the silhouette
    normalized_depth_map = depth.copy()
    normalized_depth_map[hair_silh] = (depth[hair_silh] - min_depth) / (max_depth - min_depth) 

    # Optionally, set values outside the silhouette to zero or NaN
    normalized_depth_map[~hair_silh] = 0  # or np.nan

    return normalized_depth_map


def main(args,device='cuda'):
    
    img_path = args.img_path
    hair_silh_path = args.hair_path
    
    last = img_path.split('/')[-1]
    path_to_save  = img_path.replace(last, args.prefix+last+'_aligned')
    os.makedirs(path_to_save, exist_ok=True)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    gt_img_path = args.gt_img_path
    
    img_gt = Image.open(gt_img_path)

    new_size = (512, 512)  # Replace width and height with desired dimensions
    img_gt = np.array(img_gt.resize(new_size))
    
    
    images_list = sorted(os.listdir(img_path))

    for idx, img_name in enumerate(images_list):

        pil_hair = os.path.join(hair_silh_path, images_list[idx])
        pil_img = Image.open(os.path.join(img_path, images_list[idx]))
        
        img = np.array(pil_img)

                
        if len(img.shape) > 2:
            img = img
        else:
            img = img[None]
            
        try:
            lmks_gt = fa.get_landmarks_from_image(img_gt)[0]
            lmks = fa.get_landmarks_from_image(img[:, :, :3])[0]

        except Exception as e:
            print(e,os.path.join(img_path, images_list[idx]) )
            continue
            
        matrix, _ = cv2.estimateAffinePartial2D(lmks, lmks_gt)
        aligned_image = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        
        cv2.imwrite(os.path.join(path_to_save, img_name), aligned_image[:, :, ::-1])
        
        for folder in args.all_paths_for_processing:
            try:
                if folder == 'depth_apple_pro512':
                        depth_path = os.path.join(img_path.replace(last, folder), images_list[idx].replace('.jpeg', '.npz'))
                        mask_img = Image.open(pil_hair).resize(new_size)
                        try:
                            mask_hair = np.array(mask_img)[:, :, 0] / 255.

                        except Exception as e:

                            mask_hair = np.array(mask_img) / 255.
                            
                        img_folder = resize(load_apple_pro_depth_and_normalize(depth_path, mask_hair), (512, 512), anti_aliasing=True)

                elif folder == 'confidence_maps':
                    img_folder = np.load(os.path.join(img_path.replace(last, folder), images_list[idx].split('.')[0]+'.npy'))
                    
                elif folder == 'depth_map':

                    img_folder = np.load(os.path.join(img_path.replace(last, folder), images_list[idx]).replace('jpeg','npz').replace('jpeg','')) #.split('.')[0]+'.npy'))

                elif folder == 'strand_map':
                    pil_img = Image.open(os.path.join(img_path.replace(last, folder), images_list[idx]))
                    mask_img = np.array(Image.open(pil_hair).resize(new_size)) / 255.

                    if len(mask_img.shape) > 2:
                        mask_img = mask_img[:, :, 0]
                    img_folder = np.array(pil_img) * mask_img[..., None]

                elif folder == 'orientation_maps512':
                    pil_img = Image.open(os.path.join(img_path.replace(last, folder), images_list[idx]).replace('.jpeg', '.png')).resize((512, 512))
                else:
                    pil_img = Image.open(os.path.join(img_path.replace(last, folder), images_list[idx])).resize((512, 512))

                    img_folder = np.array(pil_img)
                    if len(img_folder.shape) > 2:
                        if folder == 'seg' or folder == 'body_img':
                            img_folder = img_folder[:, :, 0] 
                        else:
                            img_folder = img_folder[:, :, :3] 

                if len(img.shape) > 2:
                    img_folder = img_folder
                else:
                    img_folder = img_folder[None]

                aligned_image_folder = cv2.warpAffine(img_folder, matrix, (img_folder.shape[1], img_folder.shape[0]))

                path_to_save_aligned_folder  = path_to_save.replace(last, folder+'_aligned')
                os.makedirs(path_to_save_aligned_folder, exist_ok=True)


                if folder == 'depth_map' or folder == 'confidence_maps':

                    np.save(os.path.join(path_to_save_aligned_folder, img_name.split('.')[0]+'.jpeg'), aligned_image_folder)

                elif folder == 'depth_apple_pro512':
                    np.savez_compressed(os.path.join(path_to_save_aligned_folder,  img_name.replace('.jpef', '.npz').replace('.jpg', '.npz')), depth=aligned_image_folder)

                else:
                    if len(aligned_image_folder.shape)> 2 and folder != 'depth_vis_map_aligned_aligned':
                        cv2.imwrite(os.path.join(path_to_save_aligned_folder, img_name), aligned_image_folder[:, :, ::-1])
                    else:
                        cv2.imwrite(os.path.join(path_to_save_aligned_folder, img_name), aligned_image_folder)
            except Exception as e:
                print(e)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--img_path', default= './dataset/resized_img', type=str)
    parser.add_argument('--hair_path', default= './dataset/seg', type=str)
    parser.add_argument('--all_paths_for_processing', nargs='+', type=str, help="List of image paths")
    parser.add_argument('--gt_img_path', default= 'inputs/data/image0021.png', type=str)
    parser.add_argument('--prefix', default= '', type=str)



    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)  
    
    
    
    
    
    