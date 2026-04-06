#!/bin/bash

set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0

dataset=gear
subset=sapien
scenes=(box_100154 bucket_100481 door_9168)
iteration=30000

voxel_size=0.01
dilation_radius=1
coarse_name=coarse_gs

python utils/voxelize_movable.py \
    --dataset "${dataset}" \
    --subset "${subset}" \
    --scenes "${scenes[@]}" \
    --iteration "${iteration}" \
    --voxel_size "${voxel_size}" \
    --dilation_radius "${dilation_radius}" \
    --coarse_name "${coarse_name}"
