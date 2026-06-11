#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4,5,6,7

SPLIT=5
DATA_PATH="${AVF_MAE_ROOT:-.}/saved/data/MAFW/audio_visual/single/split0${SPLIT}"
CHECKPOINT="${AVF_MAE_ROOT:-.}/saved/model/TEST-N-11-mix-data-109w-final-model-huge-pretrain-100eps-904/finetuning-mix-data-109w-pretrain-100eps-final-model-huge-903/MAFW/audio_visual/hicmae_pretrain_huge/checkpoint-99/eval_split05_lr_1e-3_epoch_100_size160_a256_sr4_server6000/checkpoint-best.pth"
OUTPUT_DIR="${AVF_MAE_ROOT:-.}/saved/model/TEST-N-11-mix-data-109w-final-model-huge-pretrain-100eps-904/eval-reproduce/split0${SPLIT}"
LOG_DIR="${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

python -m torch.distributed.launch --nproc_per_node=4 --master_port 13297 \
  ${AVF_MAE_ROOT:-.}/run_class_finetuning_av.py \
  --model avit_dim768_patch16_160_a256 \
  --data_set MAFW \
  --nb_classes 11 \
  --data_path "${DATA_PATH}" \
  --finetune "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --batch_size 40 \
  --num_sample 1 \
  --input_size 160 \
  --input_size_audio 256 \
  --short_side_size 160 \
  --depth 15 \
  --depth_audio 15 \
  --fusion_depth 2 \
  --num_frames 16 \
  --sampling_rate 4 \
  --opt adamw \
  --lr 1e-3 \
  --weight_decay 0.05 \
  --dist_eval \
  --test_num_segment 2 \
  --test_num_crop 2 \
  --attn_type local_global \
  --lg_region_size 2 5 10 \
  --lg_region_size_audio 4 4 \
  --lg_classify_token_type region \
  --num_workers 24 \
  --eval
