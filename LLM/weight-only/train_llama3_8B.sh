#!/bin/bash

MODEL_PATH="models/llama-3-8B"


ALPHA=0.25
BETA=0.0003

WBITS_VALUES=(3)


DATE=$(date +"%Y%m%d")


mkdir -p logs6

declare -A METHODS=(
    #["oursv2_gptqv2"]="--ours_v2 --alpha ${ALPHA} --beta ${BETA} --gptqv2"
    ["gptqv2_only"]="--gptqv2 --alpha ${ALPHA}"
    #["gptq"]=""
)

for wbits in "${WBITS_VALUES[@]}"; do
  echo "===== Testing bitwidth wbits = $wbits ====="
  
  for method_name in "${!METHODS[@]}"; do
    method_args="${METHODS[$method_name]}"
    
    echo "--- Running method: $method_name ---"
    
    LOG_FILE="logs6/${DATE}_llama3-8b-${wbits}bit-128g_${method_name}.log"
    
    CMD="CUDA_VISIBLE_DEVICES=0 python -u llama_step.py \
      $MODEL_PATH c4 \
      --wbits $wbits \
      --true-sequential \
      --act-order \
      --groupsize 128 \
      --eval \
      $method_args"
    
    echo "Executing command: $CMD"
    echo "Log will be saved to: $LOG_FILE"
    
    eval "$CMD" | tee "$LOG_FILE"
    
    echo "--- Method $method_name evaluation completed ---"
    echo ""
    
    nvidia-smi --gpu-reset 2>/dev/null || true
    sleep 10
  done
  
  echo "===== All methods for bitwidth wbits = $wbits completed ====="
  echo ""
done

echo "All tests completed!"