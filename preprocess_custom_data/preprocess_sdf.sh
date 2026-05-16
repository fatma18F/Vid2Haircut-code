#!/bin/bash

# # Check if the runid is provided as a command line argument
# if [ $# -lt 4 ]; then
#   echo "Usage: $0 <runid> <filename> <num_node> <gpu_per_node>"
#   exit 1
# fi

export CUDA_HOME=/is/software/nvidia/cuda-11.8
export LD_LIBRARY_PATH=/is/software/nvidia/cuda-11.8/lib64
export PATH=$PATH:/is/software/nvidia/cuda-11.8/bin

PYTHON_ENV=/home/vsklyarova/miniconda_latest3/bin/activate
source ${PYTHON_ENV}
conda deactivate && conda activate /home/vsklyarova/miniconda_latest3/envs/eccv_gaus_hair 

python ./preprocess_custom_data/compute_sdf.py