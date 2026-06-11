model="hicmae_pretrain_huge"
OUTPUT_DIR="./saved/model/pretraining-mix-data-109w-four-parts_final-model-huge-830-new-envs/audio_visual/${model}"
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p $OUTPUT_DIR
fi

DATA_PATH='./saved/data/mix-data/voxceleb2_CNCVS-CN-Celeb-MER24-pretrain-cropped-AVspeech-Final/audio_visual/cross-lingual-mix-data-overall-four-parts-selected-109w.csv'

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
        --model pretrain_hicmae_dim768_patch16_160_a256 \
        --encoder_depth 15 \
        --decoder_depth 4 \
        --encoder_depth_audio 15 \
        --decoder_depth_audio 4 \
        --encoder_fusion_depth 2 \
        --batch_size 100 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 5 \
        --save_ckpt_freq 10 \
        --epochs 100 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --lr 3e-4 \
        --num_workers 12 \
        --roll_mag_aug True \
        --return_intermediate_features 4 9 14 \
        --loss_weight 0.0025 \
        --inter_contrastive_temperature 0.07 \
        --attn_type local_global \
        --lg_region_size 2 5 10 \
        --lg_region_size_audio 4 4 \
        --use_frame_diff_as_target
