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
      "Multisteps"
)

folder_name="multitimesteps"


personID="412" #"186" #"186" #"412" "72"
ckpt_path="/home/ayed/3Dhairstyle/outputs/exps_person${personID}_new/singleOpt/checkpoints/ckpt_000200.pth"


for prefix in "${folders[@]}"; do
    echo "${prefix}"
    echo "__________________________"
    scene="cam_222200037_$prefix.jpg" 
    NSTEPS="-1"
    
    python mv_scanner_prior.py --conf_path ./configs/static_multi${personID}.conf \
        --savedir ./outputs/exps_person${personID}_new/${folder_name}2/ \
        --unfreeze_time_for_pca -1 \
        --num_workers 8 \
        --ckpt_path  $ckpt_path -r 1 \
        --pointcloud_path_head "./inputs/data/body_head_prior.ply" \
        --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" \
        --render_direction  \
        --binarize_masks --detect_anomaly \
        --port $PORT --ip 127.0.0.1 --scene $scene \
        --upsample_hairstyle True  --upsample_resolution $RES \
        --num_steps_coarse $NSTEPS --prefix2 $prefix  

done

