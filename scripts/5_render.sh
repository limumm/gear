export CUDA_VISIBLE_DEVICES=0
dataset=gear
subset=sapien
scenes=(box_100154 bucket_100481 door_9168)

model_name=gear
iteration=30000

for scene in ${scenes[@]};do
    model_path=outputs/${dataset}/${subset}/${scene}/${model_name}
    python render.py \
        --dataset ${dataset} \
        --subset ${subset} \
        --scene_name ${scene} \
        --model_path ${model_path} \
        --iteration ${iteration} \
        --skip_test 
done
