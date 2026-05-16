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

sys.path.append(os.path.join(sys.path[0], '..'))

sys.path.append('/home/vsklyarova/Projects/NeuralHaircut_developer')

# calc silh masks
from MODNet.src.models.modnet import MODNet
from tqdm import tqdm

import os
from copy import deepcopy


    
def main(args):

    os.makedirs(os.path.join(args.scene_path, 'hair_mask_sam'), exist_ok=True)
    
    images = sorted(os.listdir(os.path.join(args.scene_path, 'images')))
    n_images = len(sorted(os.listdir(os.path.join(args.scene_path, 'images'))))
    
    tens_list = []
    for i in range(n_images):
        tens_list.append(T.ToTensor()(Image.open(os.path.join(args.scene_path, 'images', images[i]))))

#     load MODNET model for silhouette masks
    modnet = nn.DataParallel(MODNet(backbone_pretrained=False))
    modnet.load_state_dict(torch.load(args.MODNET_ckpt))
    device = torch.device('cuda')
    modnet.eval().to(device)
    
    # Create silh masks
    silh_list = []
    for i in tqdm(range(len(tens_list))):
        silh_mask = obtain_modnet_mask(tens_list[i], modnet, 512)
        silh_list.append(silh_mask)
        cv2.imwrite(os.path.join(args.scene_path, 'body_mask', images[i]), postprocess_mask(silh_mask)[0].astype(np.uint8))
    
    print("Start calculating hair masks!")
#     load CDGNet for hair masks
    model = Res_Deeplab(num_classes=20)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    state_dict = model.state_dict().copy()
    state_dict_old = torch.load(args.CDGNET_ckpt, map_location='cpu')

    for key, nkey in zip(state_dict_old.keys(), state_dict.keys()):
        if key != nkey:
            # remove the 'module.' in the 'key'
            state_dict[key[7:]] = deepcopy(state_dict_old[key])
        else:
            state_dict[key] = deepcopy(state_dict_old[key])

    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    basenames = sorted([s.split('.')[0] for s in os.listdir(os.path.join(args.scene_path, 'images'))])
    input_size = (1024, 1024)

    raw_images = []
    images = []
    masks = []
    for basename in basenames:
        img = Image.open(os.path.join(args.scene_path, 'images', basename + '.jpg'))
        raw_images.append(np.asarray(img))
        img = transform(img.resize(input_size))[None]
        img = torch.cat([img, torch.flip(img, dims=[-1])], dim=0)
        mask = np.asarray(Image.open(os.path.join(args.scene_path, 'body_mask', basename + '.jpg')))
        images.append(img)
        masks.append(mask)

    image_size = (mask.shape[1], mask.shape[0])
    parsing_preds, hpredLst, wpredLst = valid(model, images, input_size, image_size, len(images), gpus=1)

    for i in range(len(images)):
        hair_mask = np.asarray(Image.fromarray((parsing_preds[i] == 2)).resize(image_size, Image.BICUBIC))
        hair_mask = hair_mask * masks[i]
        Image.fromarray(hair_mask).save(os.path.join(args.scene_path, 'hair_mask', basenames[i] + '.jpg'))
   
    print('Results saved in folder: ', os.path.join(args.scene_path, 'hair_mask'))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--scene_path', default='./implicit-hair-data/data/h3ds/168f8ca5c2dce5bc/', type=str)
    parser.add_argument('--MODNET_ckpt', default='/is/rg/ncs/projects/vsklyarova/processing/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt', type=str)
    parser.add_argument('--CDGNET_ckpt', default='/is/rg/ncs/projects/vsklyarova/CDGNet/LIP_epoch_149.pth', type=str)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)