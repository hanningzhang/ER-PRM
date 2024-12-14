deepspeed_args="--master_port=11110"
# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
CUDA_DEVICES=$CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
deepspeed ${deepspeed_args} \
  llama3_math_reward.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_path peiyi9979/Math-Shepherd \
    --output_dir reward_llama_31_8b --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --block_size 512 \
    --lr_scheduler_type cosine \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_step 1 \
    --gradient_checkpointing True \
    --use_flash_attention True \
    --deepspeed configs/ds_config_zero2.json \
    --bf16 \
    --run_name reward_llama_31_8b \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    # | tee ${log_dir}/train.log \
    # 2> ${log_dir}/train.err

export CUDA_VISIBLE_DEVICES=CUDA_DEVICES
