#!/bin/bash


# === Parse exp_name from command line ===
 

exp_name="exps_inverse_stage_person72" #_07_08_bestAlign"
timestep='25'

root_path="/home/ayed/monocular-hair-modeling/outputs" #"./example_data"
# exp_name="POST_ICCV_debug_sdf_multi_view_scanner_video_3d.conf_3d.conf"

blender_bin="/home/ayed/blender-3.6.0-linux-x64/blender"
blend_file="./blender_utils/render.blend"
render_script="./blender_utils/render_color_static.py"
camera_file="$root_path/$exp_name/0000$timestep/blender/selected_cam.npy"
bodys_dir="$root_path/$exp_name/0000$timestep/blender/heads"
hair_dir="$root_path/$exp_name/0000$timestep/blender"


#python ./blender_utils/convert_data_to_blender_format.py --exp_name $exp_name  --scene_path ~/monocular-hair-modeling/outputs/ --frame_id 000001 --save_dir /home/ayed/monocular-hair-modeling/outputs/exps_inverse_stage_person464/000001/blender --data_path /home/ayed/monocular-hair-modeling/inputs/example_data464/000001/ --tracker_path /home/ayed/monocular-hair-modeling/inputs/example_data464/smpl_mesh --device 'cuda'
 
for mesh_path in "$hair_dir"/*.npy; do
    filename=$(basename "$mesh_path")
    name="${filename%.*}"

    hair_path="$hair_dir/${name}.npy"

    if [ ! -f "$hair_path" ]; then
        echo "Warning: hair file $hair_path not found, skipping $mesh_path"
        continue
    fi

    echo "Rendering mesh $filename with hair $name.npy"

    "$blender_bin" -b "$blend_file" -P "$render_script" -- --args \
        --camera "$camera_file" \
        --mesh "$bodys_dir/000000.ply" \
        --hair "$hair_path" \
        --save_name "$exp_name" \
        --save_output "$hair_dir"
done
