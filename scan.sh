#!/bin/bash

# ==============================================================================
# SCRIPT QUÉT SIÊU THAM SỐ NORM_BOUND CHO NB-SINGLORA
# ==============================================================================
#
# Chức năng:
# - Chạy thử nghiệm trên bộ dữ liệu EuroSAT.
# - Lặp qua một danh sách các giá trị norm_bound để tìm ra giá trị tối ưu.
# - Tự động trích xuất accuracy cuối cùng và lưu vào tệp summary.csv.
#
# ==============================================================================

# --- PHẦN 1: CẤU HÌNH THỬ NGHIỆM ---

# Chỉ chạy trên bộ dữ liệu EuroSAT
DATASET="eurosat"

# Số lượng shots cố định
SHOT=1

# Các giá trị norm_bound cần quét
NORM_BOUNDS=(0.1 0.5 1.0 2.0 5.0 10.0) # Thêm hoặc bớt các giá trị bạn muốn thử

# Cấu hình CỐ ĐỊNH cho tất cả các lần chạy
ADAPTER_TYPE="singlora"
LEARNING_RATE="2e-4" # Sử dụng một LR đã được chứng minh là ổn định
RANK=2
ALPHA=1
RAMP_UP_STEPS=100
LORA_PARAMS="q k v" # Áp dụng cho cả MLP để thấy rõ hiệu quả
BASE_ITERS=500
BATCH_SIZE=32

# Đường dẫn
ROOT_PATH="/root/DATA"
SAVE_PATH="./checkpoints"

# Thiết lập thư mục và tệp tổng hợp
LOG_DIR="results_logs_norm_scan"
SUMMARY_FILE="results_summary_norm_scan_${DATASET}_${SHOT}shot.csv"
mkdir -p $LOG_DIR


# --- PHẦN 2: CHUẨN BỊ TỆP TỔNG HỢP ---

# Luôn tạo mới tệp summary cho mỗi lần quét siêu tham số
echo "Creating new summary file: ${SUMMARY_FILE}"
echo "Norm_Bound,Final_Accuracy,Log_File" > $SUMMARY_FILE


# --- PHẦN 3: VÒNG LẶP THỰC THI ---

# Lặp qua từng giá trị norm_bound
for NORM_BOUND in "${NORM_BOUNDS[@]}"; do

    # Tạo tên file log duy nhất cho mỗi lần chạy
    CONFIG_NAME="${ADAPTER_TYPE}_${DATASET}_${SHOT}shot_norm${NORM_BOUND}"
    LOG_FILE="${LOG_DIR}/${CONFIG_NAME}.log"
    
    # Xóa file log cũ nếu tồn tại
    rm -f $LOG_FILE

    echo "======================================================================"
    echo "Starting run: ${CONFIG_NAME}"
    echo "======================================================================"

    # Xây dựng lệnh python với norm_bound hiện tại
    COMMAND="python3 main.py \
      --dataset ${DATASET} \
      --root_path ${ROOT_PATH} \
      --shots ${SHOT} \
      --n_iters ${BASE_ITERS} \
      --batch_size ${BATCH_SIZE} \
      --adapter ${ADAPTER_TYPE} \
      --lr ${LEARNING_RATE} \
      --r ${RANK} \
      --alpha ${ALPHA} \
      --ramp_up_steps ${RAMP_UP_STEPS} \
      --norm_bound ${NORM_BOUND} \
      --params ${LORA_PARAMS} \
      --save_path ${SAVE_PATH}"

    # In lệnh ra màn hình
    echo "Executing command:"
    echo "${COMMAND}"
    echo ""

    # Thực thi lệnh và ghi output vào file log
    eval ${COMMAND} > ${LOG_FILE} 2>&1

    # --- PHẦN 4: TRÍCH XUẤT VÀ LƯU KẾT QUẢ ---

    # Tìm dòng cuối cùng chứa "Final test accuracy" trong file log
    FINAL_ACC_LINE=$(grep "Final test accuracy" "${LOG_FILE}" | tail -n 1)

    # Trích xuất số cuối cùng trong dòng
    FINAL_ACC=$(echo "$FINAL_ACC_LINE" | awk '{print $NF}' | sed 's/\.$//')

    # Nếu không tìm thấy kết quả (do lỗi), ghi là "FAILED"
    if [ -z "$FINAL_ACC" ]; then
        FINAL_ACC="FAILED"
    fi

    echo "----------------------------------------------------------------------"
    echo "Run finished for ${CONFIG_NAME}. Final Accuracy: ${FINAL_ACC}"
    echo "----------------------------------------------------------------------"
    echo ""

    # Ghi kết quả vào tệp CSV
    echo "${NORM_BOUND},${FINAL_ACC},${LOG_FILE}" >> $SUMMARY_FILE

done

echo "All experiments finished."
echo "Results summary saved to ${SUMMARY_FILE}"
echo "You can view the summary by running: cat ${SUMMARY_FILE}"