#!/bin/bash

root_path="/fast/vsklyarova/Projects/FaceContact/dataset/visualizations/touch_hair_right_hand"
exp_name="POST_ICCV_debug_sdf_multi_view_scanner_video_3d_40_steps.conf_3d.conf"

blender_bin="/home/vsklyarova/blender/blender"
blend_file="./blender_utils/render.blend"
render_script="./blender_utils/render_color.py"
camera_file="$root_path/selected_cam.npy"
bodys_dir="$root_path/bodys"
hair_dir="$root_path/$exp_name"

for mesh_path in "$bodys_dir"/*.ply; do
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
        --mesh "$mesh_path" \
        --hair "$hair_path" \
        --save_name "$exp_name"
done
