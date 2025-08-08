#!/bin/bash
# 'fgvc' 'caltech101' 'eurosat'
DATASET=('fgvc' 'caltech101' 'eurosat')
# 'ce' 'ce_ls' 'arcface' 'cosface' 'ce_center' 'ce_maxent'
LOSS_TYPES=('ce' 'ce_ls' 'arcface' 'cosface' 'ce_center' 'ce_maxent')
LOG_DIR='result_loRA'
mkdir -p "$LOG_DIR"

for dataset in "${DATASET[@]}"; do
  for loss in "${LOSS_TYPES[@]}"; do
    LOG_FILE="${LOG_DIR}/${dataset}_${loss}.log"
    rm -f "$LOG_FILE"

    echo "======================================================================"
    echo "Starting run: dataset=${dataset}, loss_type=${loss}"
    echo "======================================================================"

    COMMAND="python main.py --dataset ${dataset} --shots 1 --loss_type ${loss} --root_path /content/DATA"
    echo "Executing command:"
    echo "${COMMAND}"
    echo ""

    eval "${COMMAND}" > "${LOG_FILE}" 2>&1
  done
done
