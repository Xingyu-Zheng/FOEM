#!/bin/bash

gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id

mkdir -p logs

# ======== w4a4，beta=3e-4 ========
log_file_w4a4_beta3="logs/ours_gptq_w4a4_beta3e-4_$(date +%Y%m%d_%H%M%S).log"
echo "w4a4, beta=3e-4"
python main.py --model eva02_large_patch14_448.mim_m38m_ft_in22k_in1k  \
 --w_bits 2 \
 --w_groupsize -1 \
 --w_clip \
 --a_bits 4 \
 --nsamples 128 \
 --a_asym \
 --w_asym \
 --percdamp 0.1 \
 --act_order \
 --bsz 256 \
 --enable_aq_calibration \
 --ours \
 --beta 0.0003 \
 2>&1 | grep -v "^\r" | tee "$log_file_w4a4_beta3"

echo "w4a4, beta=3e-4 $log_file_w4a4_beta3"


# ======== w3a16，beta=3e-4 ========
log_file_w2a16_beta3="logs/ours_gptq_w2a16_beta3e-4_$(date +%Y%m%d_%H%M%S).log"
echo "w2a16, beta=3e-4"
python main.py --model eva02_large_patch14_448.mim_m38m_ft_in22k_in1k  \
 --w_bits 2 \
 --w_groupsize -1 \
 --w_clip \
 --a_bits 16 \
 --nsamples 128 \
 --a_asym \
 --w_asym \
 --percdamp 0.1 \
 --act_order \
 --bsz 256 \
 --enable_aq_calibration \
 --ours \
 --beta 0.0003 \
 2>&1 | grep -v "^\r" | tee "$log_file_w2a16_beta3"

echo "w2a16, beta=3e-4 $log_file_w2a16_beta3"
