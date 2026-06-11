#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1

ROOT="${AVF_MAE_ROOT:-.}"
OLD_PREFIX="${OLD_MAFW_DATA_ROOT:-./datasets/MAFW/data}"
NEW_PREFIX="${MAFW_DATA_ROOT:-./datasets/MAFW/data}"
CHECKPOINT_BASE="${ROOT}/saved/model/TEST-N-11-mix-data-109w-final-model-huge-pretrain-100eps-904/finetuning-mix-data-109w-pretrain-100eps-final-model-huge-903/MAFW/audio_visual/hicmae_pretrain_huge/checkpoint-99"
OUTPUT_ROOT="${ROOT}/saved/model/TEST-N-11-mix-data-109w-final-model-huge-pretrain-100eps-904/eval-reproduce-all-splits"
SCRIPT="${ROOT}/run_class_finetuning_av.py"

mkdir -p "${OUTPUT_ROOT}"

RESULTS_FILE="${OUTPUT_ROOT}/results.txt"
echo "MAFW 5-fold evaluation results - $(date)" > "${RESULTS_FILE}"
echo "----------------------------------------" >> "${RESULTS_FILE}"

declare -a uar_values
declare -a war_values

for split in 2 3 4 5; do
  echo "============================================"
  echo "Evaluating split0${split}"
  echo "============================================"

  SOURCE_SPLIT_DIR="${ROOT}/saved/data/MAFW/audio_visual/single/split0${split}"
  TARGET_SPLIT_DIR="${ROOT}/saved/data/MAFW/audio_visual/single/split0${split}_local"
  CHECKPOINT="${CHECKPOINT_BASE}/eval_split0${split}_lr_1e-3_epoch_100_size160_a256_sr4_server6000/checkpoint-best.pth"
  OUTPUT_DIR="${OUTPUT_ROOT}/split0${split}"
  LOG_FILE="${OUTPUT_DIR}/log.txt"

  mkdir -p "${TARGET_SPLIT_DIR}"
  mkdir -p "${OUTPUT_DIR}"

  for f in train.csv test.csv; do
    if [ -f "${SOURCE_SPLIT_DIR}/${f}" ]; then
      sed "s|${OLD_PREFIX}|${NEW_PREFIX}|g" "${SOURCE_SPLIT_DIR}/${f}" > "${TARGET_SPLIT_DIR}/${f}"
    fi
  done

  if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint not found for split0${split}: ${CHECKPOINT}"
    exit 1
  fi

  python -m torch.distributed.launch --nproc_per_node=1 --master_port 13297 \
    "${SCRIPT}" \
    --model avit_dim768_patch16_160_a256 \
    --data_set MAFW \
    --nb_classes 11 \
    --data_path "${TARGET_SPLIT_DIR}" \
    --finetune "${CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${OUTPUT_DIR}" \
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
    --eval | tee "${OUTPUT_DIR}/run.log"

  if [ -f "${LOG_FILE}" ]; then
    uar=$(grep -oE 'UAR: [0-9]+\.[0-9]+%' "${LOG_FILE}" | tail -n 1 | sed 's/UAR: \([0-9.]*\)%/\1/')
    war=$(grep -oE 'WAR: [0-9]+\.[0-9]+%' "${LOG_FILE}" | tail -n 1 | sed 's/WAR: \([0-9.]*\)%/\1/')
  else
    uar=$(grep -oE 'UAR: [0-9]+\.[0-9]+%' "${OUTPUT_DIR}/run.log" | tail -n 1 | sed 's/UAR: \([0-9.]*\)%/\1/')
    war=$(grep -oE 'WAR: [0-9]+\.[0-9]+%' "${OUTPUT_DIR}/run.log" | tail -n 1 | sed 's/WAR: \([0-9.]*\)%/\1/')
  fi

  if [ -z "${uar}" ] || [ -z "${war}" ]; then
    echo "WARNING: Could not extract UAR/WAR for split0${split}."
    continue
  fi

  echo "split0${split} UAR=${uar}% WAR=${war}%"
  uar_values+=("${uar}")
  war_values+=("${war}")
  echo "split0${split} UAR=${uar}% WAR=${war}%" >> "${RESULTS_FILE}"

done

if [ ${#uar_values[@]} -eq 0 ]; then
  echo "No UAR/WAR values were collected. Exiting."
  exit 1
fi

sum_uar=0
sum_war=0
for v in "${uar_values[@]}"; do
  sum_uar=$(awk "BEGIN {printf \"%.6f\", ${sum_uar} + ${v}}")
done
for v in "${war_values[@]}"; do
  sum_war=$(awk "BEGIN {printf \"%.6f\", ${sum_war} + ${v}}")
done

num=${#uar_values[@]}
avg_uar=$(awk "BEGIN {printf \"%.2f\", ${sum_uar} / ${num}}")
avg_war=$(awk "BEGIN {printf \"%.2f\", ${sum_war} / ${num}}")

echo "============================================"
echo "Average UAR over ${num} splits: ${avg_uar}%"
echo "Average WAR over ${num} splits: ${avg_war}%"
echo "============================================"
echo "" >> "${RESULTS_FILE}"
echo "Average UAR over ${num} splits: ${avg_uar}%" >> "${RESULTS_FILE}"
echo "Average WAR over ${num} splits: ${avg_war}%" >> "${RESULTS_FILE}"
echo "============================================" >> "${RESULTS_FILE}"
