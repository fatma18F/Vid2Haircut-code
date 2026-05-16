import cv2 as cv
import os 
from torchvision import transforms
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import argparse
from tqdm import tqdm
import sys
sys.path.append(os.path.join(sys.path[0], './utils'))

import torch.nn.functional as F
sys.path.append('/home/vsklyarova/Projects/VAE_channel_reduction/Baselines')


from gabor_filter import GaborFilter




def main(args,device='cuda'):
    
    img_path = args.img_path
    path_to_save  = args.path_to_save
    path_to_save_conf = args.path_to_save_conf

    os.makedirs(path_to_save, exist_ok=True)
    os.makedirs(path_to_save_conf, exist_ok=True)
    
    gb = GaborFilter()


    images_list = os.listdir(img_path)


    for idx, img_name in tqdm(enumerate(images_list)):

        pil_img = Image.open(os.path.join(img_path, images_list[idx]))
        img = (torch.tensor(np.array(pil_img)).cuda() / 255.).permute(2, 0, 1)[None]

        pil_img = Image.open(os.path.join(img_path.replace('images', 'sam_hair_mask'), images_list[idx]))
        
        print((torch.tensor(np.array(pil_img)).cuda() / 255.).shape)
        try:
            mask = (torch.tensor(np.array(pil_img)).cuda() / 255.)[:, :, 0][None]
        except Exception as e:
            mask = (torch.tensor(np.array(pil_img)).cuda() / 255.)[None]

        ormaps, confidence_map = gb(img)

        ormaps = (mask * ormaps[0].cuda()).clone()                  

        cv2.imwrite(os.path.join(path_to_save, images_list[idx]), ormaps[0].detach().cpu().numpy())  
        
        np.save(os.path.join(path_to_save_conf, images_list[idx]).replace('png', 'npy').replace('jpg', 'npy'), confidence_map[0].numpy().astype('float16'))                     
        
        

        
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--img_path', default= '/home/vsklyarova/Projects/VAE_channel_reduction/Baselines/Validation_CVPR/dataset/resized_img', type=str)

    parser.add_argument('--path_to_save', default= '/home/vsklyarova/Projects/VAE_channel_reduction/Baselines/Validation_CVPR/dataset/orientation_maps', type=str)
    
    parser.add_argument('--path_to_save_conf', default= '/home/vsklyarova/Projects/VAE_channel_reduction/Baselines/Validation_CVPR/dataset/confidence_maps', type=str)


    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)  
    
    
    
    
    
    