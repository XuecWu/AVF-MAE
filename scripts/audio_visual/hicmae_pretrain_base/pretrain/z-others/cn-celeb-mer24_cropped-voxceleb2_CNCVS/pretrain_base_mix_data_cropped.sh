model="hicmae_pretrain_base"
OUTPUT_DIR="./saved/model/pretraining/mix_data-701/audio_visual/${model}"
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p $OUTPUT_DIR
fi
DATA_PATH='./saved/data/CN-Celeb-MER24-cropped_voxceleb2_CNCVS/CN-Celeb-MER24-cropped_voxceleb2_CNCVS_av_pretrain-new.csv'
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 \
        --master_port 11120 \
        run_mae_pretraining_av.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --input_size 160 \
        --mask_ratio_audio 0.8125 \
        --input_size_audio 256 \
        --model pretrain_hicmae_dim512_patch16_160_a256 \
        --encoder_depth 10 \
        --decoder_depth 4 \
        --encoder_depth_audio 10 \
        --decoder_depth_audio 4 \
        --encoder_fusion_depth 2 \
        --batch_size 32 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 5 \
        --save_ckpt_freq 2 \
        --epochs 100 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --lr 3e-4 \
        --num_workers 16 \
        --roll_mag_aug True \
        --return_intermediate_features 3 6 9 \
        --loss_weight 0.0025 \
        --inter_contrastive_temperature 0.07 \
        --attn_type local_global \
        --lg_region_size 2 5 10 \
        --lg_region_size_audio 4 4 \
        --use_frame_diff_as_target \
