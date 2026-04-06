#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

dataset=gear
subset=sapien
scenes=(box_100154 bucket_100481 clock_6843)

for scene in "${scenes[@]}"; do
    dataset_path="${dataset}/${subset}/${scene}"
    python utils/get_mask.py --dataset_path "${dataset_path}"
done