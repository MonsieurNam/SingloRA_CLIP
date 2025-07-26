#!/bin/bash

# ==============================================================================
# SCRIPT TỰ ĐỘNG CHẠY VÀ TỔNG HỢP KẾT QUẢ
# ==============================================================================

# --- PHẦN 1: CẤU HÌNH THỬ NGHIỆM ---

DATASETS=('fgvc' 'dtd' 'eurosat' 'caltech101' 'food101' 'oxford_flowers' 'ucf101' 'oxford_pets')
SHOTS=(1)

# Cấu hình CỐ ĐỊNH
ADAPTER_TYPE="mhsinglora"
LEARNING_RATE="2e-4"
RANK=4
ALPHA=1
RAMP_UP_STEPS=100
LORA_PARAMS="q k v"
BASE_ITERS=500
BATCH_SIZE=32
NUM_HEAD_OPTIONS=(1 2 4)

ROOT_PATH="/root/DATA"
SAVE_PATH="./checkpoints"

# Tạo thư mục log nếu chưa tồn tại
LOG_DIR="results_logs_${ADAPTER_TYPE}"
mkdir -p $LOG_DIR

# --- PHẦN 3: VÒNG LẶP THỰC THI ---

for DATASET in "${DATASETS[@]}"; do
  for SHOT in "${SHOTS[@]}"; do
    for NUM_HEADS in "${NUM_HEAD_OPTIONS[@]}"; do

      CONFIG_NAME="${ADAPTER_TYPE}_${DATASET}_${SHOT}shot_heads${NUM_HEADS}"
      LOG_FILE="${LOG_DIR}/${CONFIG_NAME}.log"

      rm -f $LOG_FILE

      echo "======================================================================"
      echo "Starting run: ${CONFIG_NAME}"
      echo "======================================================================"

      COMMAND="python3 main.py \
        --dataset ${DATASET} \
        --root_path ${ROOT_PATH} \
        --shots ${SHOT} \
        --n_iters ${BASE_ITERS} \
        --num_heads ${NUM_HEADS} \
        --batch_size ${BATCH_SIZE} \
        --adapter ${ADAPTER_TYPE} \
        --lr ${LEARNING_RATE} \
        --r ${RANK} \
        --alpha ${ALPHA} \
        --ramp_up_steps ${RAMP_UP_STEPS} \
        --params ${LORA_PARAMS} \
        --save_path ${SAVE_PATH}"

      echo "Executing command:"
      echo "${COMMAND}"
      echo ""

      eval ${COMMAND} > ${LOG_FILE} 2>&1

    done
  done
done

echo "All experiments finished."
