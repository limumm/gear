export CUDA_VISIBLE_DEVICES=1


dataset=gear
subset=sapien
scenes=(box_100154 bucket_100481 door_9168)


seed=0
model_name=gear
coarse_iter=30000

# train
for scene in ${scenes[@]};do
    model_path=outputs/${dataset}/${subset}/${scene}/${model_name}
    python train.py \
        --dataset ${dataset} \
        --subset ${subset} \
        --scene_name ${scene} \
        --model_path ${model_path} \
        --eval \
        --iterations 30000 \
        --coarse_name coarse_gs \
        --coarse_iter 30000 \
        --seed ${seed} \
        --random_bg_color \
        --densify_from_iter 3000 \
        --densify_grad_threshold 0.001 \
        
done

# test
for scene in ${scenes[@]};do
    model_path=outputs/${dataset}/${subset}/${scene}/${model_name}
    python render.py \
        --dataset ${dataset} \
        --subset ${subset} \
        --scene_name ${scene} \
        --model_path ${model_path} \
        --resolution ${res} \
        --iteration best \
        --skip_test 
done
