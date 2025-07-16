
base_dir="logs"
script_dir="llama3-8B-ours_gptqv2"
mkdir -p $base_dir

for w in 4
do
    for a in 4
    do
        kv_bit=4
        log_dir="$base_dir/$script_dir/w${w}/a${a}/kv${kv_bit}"
        mkdir -p $log_dir
        log_file="$log_dir/W${w}A${a}KV${kv_bit}alpha0.30beta0.0001_ours_gptqv2_rotate.log"
        
        CUDA_VISIBLE_DEVICES="0" torchrun --nnodes=1 --nproc_per_node=1 --master_port 32345 ptq.py \
        --input_model "models/llama3-8b" \
        --do_train False \
        --do_eval True \
        --per_device_eval_batch_size 16 \
        --model_max_length 2048 \
        --fp16 False \
        --bf16 True \
        --save_safetensors False \
        --w_bits ${w} \
        --a_bits ${a} \
        --k_bits ${kv_bit} \
        --v_bits ${kv_bit} \
        --w_clip \
        --a_asym \
        --k_asym \
        --v_asym \
        --k_groupsize 128 \
        --v_groupsize 128 \
        --rotate \
        --optimized_rotation_path "SpinQuant/output_rotation/LLaMA-3-8B/8B_W16A4KV4_lr_1.5_seed_0/R.bin" \
        --use_v2 \
        --alpha 0.30 \
        --beta 0.0001 \
        --ours_v2 >> $log_file
    done
done

