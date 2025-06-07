server=6000
pretrain_dataset='final-model-huge-903'
# dataset
finetune_dataset='MAFW'
num_labels=11
# model
model_dir="pretrain_huge"
ckpts=(99)
# input
input_size=160
input_size_audio=256
sr=4
# parameter
lr=1e-3
epochs=100

splits=(1 2 3 4 5)

for split in "${splits[@]}";
do
  for ckpt in "${ckpts[@]}";
  do
    # output directory
    OUTPUT_DIR="./saved/model/finetuning-${pretrain_dataset}/${finetune_dataset}/audio_visual/${model_dir}/checkpoint-${ckpt}/eval_split0${split}_lr_${lr}_epoch_${epochs}_size${input_size}_a${input_size_audio}_sr${sr}_server${server}"
    if [ ! -d "$OUTPUT_DIR" ]; then
      mkdir -p $OUTPUT_DIR
    fi
    DATA_PATH="./saved/data/${finetune_dataset}/audio_visual/single/split0${split}"
    MODEL_PATH="./saved/model/pretraining-mix-data/audio_visual/${model_dir}/checkpoint-${ckpt}.pth"

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 \
        --master_port 13297 \
        run_class_finetuning_av.py \
        --model avit_dim768_patch16_160_a256 \
        --data_set ${finetune_dataset^^} \
        --nb_classes ${num_labels} \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 40 \
        --num_sample 1 \
        --input_size ${input_size} \
        --input_size_audio ${input_size_audio} \
        --short_side_size ${input_size} \
        --depth 15 \
        --depth_audio 15 \
        --fusion_depth 2 \
        --save_ckpt_freq 1000 \
        --num_frames 16 \
        --sampling_rate ${sr} \
        --opt adamw \
        --lr ${lr} \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs ${epochs} \
        --dist_eval \
        --test_num_segment 2 \
        --test_num_crop 2 \
        --attn_type local_global \
        --lg_region_size 2 5 10 \
        --lg_region_size_audio 4 4 \
        --lg_classify_token_type region \
        --num_workers 24

  done
done
echo "Done!"