export CUDA_VISIBLE_DEVICES=2
# dataset=paris
# subset=sapien
# scenes=(foldchair_102255 washer_103776 fridge_10905 blade_103706 storage_45135 oven_101917 stapler_103111 USB_100109 laptop_10211 scissor_11100)
# subset=realscan
# scenes=(real_fridge real_storage)

# dataset=dta
# subset=sapien
# scenes=(fridge_10489 storage_47254)

dataset=gear
subset=sapien
scenes=(box_100154 bucket_100481 clock_6843 door_9168 eyeglasses_101284 faucet_1028 knife_101068oven_7187 storage_45271)

model_name=coarse_gs
for scene in ${scenes[@]};do
    model_path=outputs/${dataset}/${subset}/${scene}/${model_name}
    python train_coarse.py \
        --dataset ${dataset} \
        --subset ${subset} \
        --scene_name ${scene} \
        --model_path ${model_path} \
        --iterations 30000 \
        --opacity_reg_weight 0.1 \
        --random_bg_color \
        --init_from_pcd
done
