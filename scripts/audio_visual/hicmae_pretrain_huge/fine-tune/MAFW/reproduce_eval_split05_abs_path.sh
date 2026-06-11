#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1

SPLIT=5
SOURCE_SPLIT_DIR="${AVF_MAE_ROOT:-.}/saved/data/MAFW/audio_visual/single/split0${SPLIT}"
TARGET_SPLIT_DIR="${AVF_MAE_ROOT:-.}/saved/data/MAFW/audio_visual/single/split0${SPLIT}_local"
OLD_PREFIX="${OLD_MAFW_DATA_ROOT:-./datasets/MAFW/data}"
NEW_PREFIX="${MAFW_DATA_ROOT:-./datasets/MAFW/data}"

CHECKPOINT="${AVF_MAE_ROOT:-.}/saved/model/TEST-N-11-mix-data-109w-final-model-huge-pretrain-100eps-904/finetuning-mix-data-109w-pretrain-100eps-final-model-huge-903/MAFW/audio_visual/hicmae_pretrain_huge/checkpoint-99/eval_split05_lr_1e-3_epoch_100_size160_a256_sr4_server6000/checkpoint-best.pth"
OUTPUT_DIR="${AVF_MAE_ROOT:-.}/saved/model/TEST-N-11-mix-data-109w-final-model-huge-pretrain-100eps-904/eval-reproduce/split0${SPLIT}"
LOG_DIR="${OUTPUT_DIR}"

mkdir -p "${TARGET_SPLIT_DIR}"
mkdir -p "${OUTPUT_DIR}"

for f in train.csv test.csv; do
  if [ -f "${SOURCE_SPLIT_DIR}/${f}" ]; then
    sed "s|${OLD_PREFIX}|${NEW_PREFIX}|g" "${SOURCE_SPLIT_DIR}/${f}" > "${TARGET_SPLIT_DIR}/${f}"
  fi
 done

python -m torch.distributed.launch --nproc_per_node=1 --master_port 13297 \
  ${AVF_MAE_ROOT:-.}/run_class_finetuning_av.py \
  --model avit_dim768_patch16_160_a256 \
  --data_set MAFW \
  --nb_classes 11 \
  --data_path "${TARGET_SPLIT_DIR}" \
  --finetune "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --batch_size 1 \
  --num_sample 1 \
  --input_size 160 \
  --input_size_audio 256 \
  --short_side_size 160 \
  --depth 15 \
  --depth_audio 15 \
  --fusion_depth 2 \
  --save_ckpt_freq 1000 \
  --num_frames 16 \
  --sampling_rate 4 \
  --opt adamw \
  --lr 1e-3 \
  --opt_betas 0.9 0.999 \
  --weight_decay 0.05 \
  --epochs 100 \
  --dist_eval \
  --test_num_segment 2 \
  --test_num_crop 2 \
  --attn_type local_global \
  --lg_region_size 2 5 10 \
  --lg_region_size_audio 4 4 \
  --lg_classify_token_type region \
  --num_workers 4 \
  --eval
