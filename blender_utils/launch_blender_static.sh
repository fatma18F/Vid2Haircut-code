#!/bin/bash


# === Parse exp_name from command line ===
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <exp_name>"
    exit 1
fi

exp_name="$1"


root_path="./example_data"
# exp_name="POST_ICCV_debug_sdf_multi_view_scanner_video_3d.conf_3d.conf"

blender_bin="/home/vsklyarova/blender/blender"
blend_file="./blender_utils/render.blend"
render_script="./blender_utils/render_color_static.py"
camera_file="$root_path/selected_cam.npy"
bodys_dir="$root_path/bodys"
hair_dir="$root_path/$exp_name"


python ./blender_utils/convert_data_to_blender_format.py --exp_name $exp_name --scene_path /home/vsklyarova/Projects/monocular-hair-modeling/exps_inverse_stage 

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
        --save_name "$exp_name"
done
