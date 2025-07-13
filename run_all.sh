#!/bin/bash

# ==============================================================================
# SCRIPT TỰ ĐỘNG CHẠY FINE-TUNING CLIP VỚI LORA/SINGLORA
# ==============================================================================
#
# Hướng dẫn sử dụng:
# 1. Lưu tệp này với tên 'run.sh' trong thư mục /root/CLIP-LoRA/
# 2. Cấp quyền thực thi cho tệp: chmod +x run.sh
# 3. Chạy script từ terminal: ./run.sh
#
# Script sẽ tự động lặp qua danh sách các bộ dữ liệu và số lượng shots,
# sau đó gọi lệnh python main.py với các cấu hình tương ứng.
#
# ==============================================================================

DATASETS=("caltech101" "dtd" "eurosat" "food101" "oxford_pets" "oxford_flowers" "ucf101" "fgvc")

# Số lượng shots cần chạy (thêm hoặc bớt số trong dấu ngoặc)
SHOTS=(1 4)

# Loại adapter cần sử dụng ('lora' hoặc 'singlora')
ADAPTER_TYPE="singlora"

# Đường dẫn đến thư mục chứa dữ liệu
ROOT_PATH="/root/DATA"

# Đường dẫn để lưu các checkpoint
SAVE_PATH="./checkpoints"

# Siêu tham số dành riêng cho SingLoRA
RAMP_UP_STEPS=100

# Thư mục để lưu file log kết quả
LOG_DIR="results_logs"
mkdir -p $LOG_DIR


# --- PHẦN 2: VÒNG LẶP THỰC THI ---

# Lặp qua từng bộ dữ liệu
for DATASET in "${DATASETS[@]}"; do
  # Lặp qua từng số lượng shots
  for SHOT in "${SHOTS[@]}"; do

    # Tính toán tổng số lần lặp
    TOTAL_ITERS=$((500 * $SHOT))

    # Tạo tên file log duy nhất cho mỗi lần chạy
    LOG_FILE="${LOG_DIR}/${ADAPTER_TYPE}_${DATASET}_${SHOT}shot_r${RANK}_a${ALPHA}_lr${LEARNING_RATE}.log"

    echo "======================================================================" | tee -a $LOG_FILE
    echo "Starting run for: ${DATASET} with ${SHOT} shots" | tee -a $LOG_FILE
    echo "Adapter: ${ADAPTER_TYPE}, Rank: ${RANK}, Alpha: ${ALPHA}, LR: ${LEARNING_RATE}" | tee -a $LOG_FILE
    echo "Saving logs to: ${LOG_FILE}"
    echo "======================================================================" | tee -a $LOG_FILE

    # Xây dựng lệnh python
    COMMAND="python3 main.py \
      --dataset ${DATASET} \
      --root_path ${ROOT_PATH} \
      --shots ${SHOT} \
      --adapter ${ADAPTER_TYPE} \
      --ramp_up_steps ${RAMP_UP_STEPS} \
      --save_path ${SAVE_PATH}"

    # In lệnh ra màn hình
    echo "Executing command:"
    echo "${COMMAND}"
    echo ""

    # Thực thi lệnh và ghi output vào cả terminal và file log
    # stdbuf -oL -eL giúp ghi log theo thời gian thực thay vì đợi buffer đầy
    stdbuf -oL -eL ${COMMAND} | tee -a $LOG_FILE

    echo "Finished run for: ${DATASET} with ${SHOT} shots." | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE

  done
done

echo "All experiments finished."