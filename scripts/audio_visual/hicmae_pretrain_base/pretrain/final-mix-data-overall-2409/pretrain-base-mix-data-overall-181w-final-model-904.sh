model="hicmae_pretrain_base"
OUTPUT_DIR="./saved/model/pretraining-base-mix-data-overall-181w-final-model-904/audio_visual/${model}"
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p $OUTPUT_DIR
fi

DATA_PATH='./saved/data/mix-data-real-final-2409/audio_visual/real-final-overall-181w-2409.csv'

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
        --master_port 17328 \
        run_mae_pretraining_av.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --input_size 160 \
        --mask_ratio_audio 0.8125 \
        --input_size_audio 256 \
        --decoder_mask_type run_cell \
        --decoder_mask_ratio 0.5 \
        --decoder_mask_type_audio random \
        --decoder_mask_ratio_audio 0.5 \
        --model pretrain_hicmae_dim512_patch16_160_a256 \
        --encoder_depth 10 \
        --decoder_depth 4 \
        --encoder_depth_audio 10 \
        --decoder_depth_audio 4 \
        --encoder_fusion_depth 2 \
        --batch_size 164 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 20 \
        --save_ckpt_freq 20 \
        --epochs 200 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --lr 3e-4 \
        --num_workers 12 \
        --roll_mag_aug True \
        --return_intermediate_features 3 6 9 \
        --loss_weight 0.0025 \
        --inter_contrastive_temperature 0.07 \
        --attn_type local_global \
        --lg_region_size 2 5 10 \
        --lg_region_size_audio 4 4 \
        --use_frame_diff_as_target
