#!/bin/bash

# ==============================================================================
# SCRIPT TỰ ĐỘNG CHẠY VÀ TỔNG HỢP KẾT QUẢ
# ==============================================================================
#
# Chức năng:
# - Lặp qua các bộ dữ liệu và số lượng shots được chỉ định.
# - Chạy thử nghiệm với một bộ siêu tham số CỐ ĐỊNH.
#
# ==============================================================================

# --- PHẦN 1: CẤU HÌNH THỬ NGHIỆM ---
# 'dtd', 'eurosat', 'caltech101', 'food101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 'sun397', 'ucf101', 'imagenet', 'fgvc'

# Các bộ dữ liệu cần chạy
DATASETS=('fgvc' 'dtd' 'eurosat' 'caltech101' 'food101' 'oxford_flowers' 'ucf101' 'oxford_pets' )

# Số lượng shots cần chạy
SHOTS=(1)

# Cấu hình CỐ ĐỊNH cho tất cả các lần chạy
ADAPTER_TYPE="mhsinglora"
LEARNING_RATE="2e-4"
RANK=2
ALPHA=1
RAMP_UP_STEPS=100
LORA_PARAMS="q k v"
BASE_ITERS=500
BATCH_SIZE=32
NUM_HEADS=2
# Đường dẫn
ROOT_PATH="/root/DATA"
SAVE_PATH="./checkpoints"

# Thiết lập thư mục và tệp tổng hợp
LOG_DIR="results_logs_${ADAPTER_TYPE}"
mkdir -p $LOG_DIR

# --- PHẦN 3: VÒNG LẶP THỰC THI ---

# Lặp qua từng bộ dữ liệu
for DATASET in "${DATASETS[@]}"; do
  # Lặp qua từng số lượng shots
  for SHOT in "${SHOTS[@]}"; do

    # Tạo tên file log duy nhất cho mỗi lần chạy
    CONFIG_NAME="${ADAPTER_TYPE}_${DATASET}_${SHOT}shot"
    LOG_FILE="${LOG_DIR}/${CONFIG_NAME}.log"
    
    # Xóa file log cũ để đảm bảo log mới sạch sẽ
    rm -f $LOG_FILE

    echo "======================================================================"
    echo "Starting run: ${CONFIG_NAME}"
    echo "======================================================================"

    # Xây dựng lệnh python với các cấu hình đã định sẵn
    COMMAND="python3 main.py \
      --dataset ${DATASET} \
      --root_path ${ROOT_PATH} \
      --shots ${SHOT} \
      --n_iters ${BASE_ITERS} \
      --num_heads ${NUM_HEADS}\
      --batch_size ${BATCH_SIZE} \
      --adapter ${ADAPTER_TYPE} \
      --lr ${LEARNING_RATE} \
      --r ${RANK} \
      --alpha ${ALPHA} \
      --ramp_up_steps ${RAMP_UP_STEPS} \
      --params ${LORA_PARAMS} \
      --save_path ${SAVE_PATH}"

    # In lệnh ra màn hình
    echo "Executing command:"
    echo "${COMMAND}"
    echo ""

    # Thực thi lệnh và ghi toàn bộ output (cả stdout và stderr) vào file log
    eval ${COMMAND} > ${LOG_FILE} 2>&1

    # --- PHẦN 4: TRÍCH XUẤT VÀ LƯU KẾT QUẢ ---

    # Tìm dòng cuối cùng chứa "Final test accuracy" trong file log
    FINAL_ACC_LINE=$(grep "Final test accuracy" "${LOG_FILE}" | tail -n 1)

    # Trích xuất số cuối cùng trong dòng, loại bỏ dấu chấm nếu có
    FINAL_ACC=$(echo "$FINAL_ACC_LINE" | awk '{print $NF}' | sed 's/[[:punct:]]//g')

    # Nếu không tìm thấy kết quả (do lỗi), ghi là "FAILED"
    if [ -z "$FINAL_ACC" ]; then
        FINAL_ACC="FAILED"
    fi

    echo "----------------------------------------------------------------------"
    echo "Run finished for ${CONFIG_NAME}. Final Accuracy: ${FINAL_ACC}"
    echo "----------------------------------------------------------------------"
    echo ""

  done
done

echo "All experiments finished."