#!/bin/bash
# Dataset list
DATASET=('fgvc' 'dtd' 'eurosat')

# Loss types
LOSS_TYPES=('ce' 'ce_ls' 'arcface' 'cosface' 'ce_center' 'ce_maxent')

# Shots
SHOTS=(1 4 16)

# Log directory
LOG_DIR='result_loRA'
mkdir -p "$LOG_DIR"

for dataset in "${DATASET[@]}"; do
  for loss in "${LOSS_TYPES[@]}"; do
    for shot in "${SHOTS[@]}"; do
      LOG_FILE="${LOG_DIR}/${dataset}_${loss}_shot${shot}.log"
      rm -f "$LOG_FILE"

      echo "======================================================================"
      echo "Starting run: dataset=${dataset}, loss_type=${loss}, shots=${shot}"
      echo "======================================================================"

      COMMAND="python3 main.py --dataset ${dataset} --shots ${shot} --loss_type ${loss} --root_path /root/DATA"
      echo "Executing command:"
      echo "${COMMAND}"
      echo ""

      eval "${COMMAND}" > "${LOG_FILE}" 2>&1
    done
  done
done
