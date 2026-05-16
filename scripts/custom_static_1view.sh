#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

PROJECT_DIR="/home/ayed/3Dhairstyle/submodules/gaussian-splatting-hair"
conda deactivate &&  conda activate gaussian_splatting_hair 
 
random_number=$((1000 + RANDOM % 5001))
random_number2=$((1000 + RANDOM % 5002))
PORT="$random_number"
IP="$random_number2"


RES="256"
NSTEPS="20"

folders=(
     "000000" #"Multisteps" #"000020"  "000030"  "000040"   "000050"   "000060"   "000070"  "000080"  #"Multisteps" 
)

folder_name="singleOpt2"


ckpt_path="./pretrained_models/fine.pth"

personID="186" #"186" #"214" 

for prefix in "${folders[@]}"; do
    echo "${prefix}"
    echo "__________________________"
    scene="cam_222200037_$prefix.jpg" 
    #NSTEPS="-1"
    NSTEPS="20"

    
    python mv_scanner_prior.py --conf_path ./configs/static_1view${personID}.conf \
        --savedir ./outputs/exps_person${personID}_new/${folder_name}/ \
        --unfreeze_time_for_pca -1 \
        --num_workers 1 \
        --ckpt_path  $ckpt_path -r 1 \
        --pointcloud_path_head "./inputs/data/body_head_prior.ply" \
        --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" \
        --render_direction  \
        --binarize_masks --detect_anomaly \
        --port $PORT --ip 127.0.0.1 --scene $scene \
        --upsample_hairstyle True  --upsample_resolution $RES \
        --num_steps_coarse $NSTEPS --prefix2 $prefix  

done

