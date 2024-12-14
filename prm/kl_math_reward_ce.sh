deepspeed_args="--master_port=11111"
# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
CUDA_DEVICES=$CUDA_VISIBLE_DEVICES
#export CUDA_VISIBLE_DEVICES=0,1,6,7
deepspeed --include localhost:0,1,2,3,4,5,6,7 ${deepspeed_args} \
  kl_math_reward_ce.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_path autoregressive_reward_data_mix_best.json \
    --output_dir reward_llama_31_8b_autoregressive_mix_best --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 5e-7 \
    --block_size 512 \
    --lr_scheduler_type cosine \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_step 1 \
    --gradient_checkpointing True \
    --use_flash_attention True \
    --deepspeed ../configs/ds_config_zero2.json \
    --bf16 \
    --run_name reward_llama_31_8b_mix_best \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 500000 \
    --save_only_model True \
    --dataloader_num_workers 1 \
    # | tee ${log_dir}/train.log \
    # 2> ${log_dir}/train.err

export CUDA_VISIBLE_DEVICES=CUDA_DEVICES
