deepspeed_args="--master_port=11100"
deepspeed --include localhost:$7 ${deepspeed_args} \
  sft.py \
    --model_name_or_path $1 \
    --dataset_path $3 \
    --output_dir $2 --overwrite_output_dir \
    --current_iteration $4 \
    --total_iteration $5 \
    --random_seed $6 \
    --num_train_epochs 2 \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_step 1 \
    --gradient_checkpointing True \
    --use_flash_attention True \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --bf16 True \
    --run_name iter_$4 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --save_strategy no \
    --dataloader_num_workers 1 \
    # | tee ${log_dir}/train.log \
    # 2> ${log_dir}/train.err

