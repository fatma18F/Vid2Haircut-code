#!/bin/bash




ids=('Bob' 'Wavy' 'MopTop' 'Wings' )  # add more ids here
#ids=( 140)
for id in "${ids[@]}"; do
    echo "Processing person${id} ..."

    #python                       calc_alignment.py --img_path $path/resized_img --hair_path $path/seg --all_paths_for_processing   seg body_img orientation_maps depth_apple_pro strand_map
    #python preprocess_custom_data/calc_alignment.py --img_path $path/resized_img --hair_path $path/hair_mask512 --all_paths_for_processing   hair_mask512 body_mask512 orientation_maps512 depth_apple_pro512 strand_map512
    
    # Multisteps
    base="/home/ayed/prerocessing_input_monocular_hair_modelling/sim_data/data/${id}"
    python preprocess_custom_data/calc_alignment.py \
        --img_path "${base}/resized_img" \
        --hair_path "${base}/hair_mask512" \
        --all_paths_for_processing \
        hair_mask512 body_mask512    orientation_maps512 
    
done


