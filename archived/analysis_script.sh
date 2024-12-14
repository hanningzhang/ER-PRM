deepspeed_args="--master_port=11110"
# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
CUDA_DEVICES=$CUDA_VISIBLE_DEVICES
#export CUDA_VISIBLE_DEVICES=0,1,6,7
deepspeed --include localhost:0,1,2,3,4,5,6,7 ${deepspeed_args} \
  kl_llama3_math_reward_ce.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_path autoregressive_reward_data_co10_gsm.json \
    --output_dir reward_llama_31_8b_autoregressive_ce10_gsm --overwrite_output_dir \
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
    --run_name reward_llama_31_8b_ce5 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 500000 \
    --save_only_model True \
    --dataloader_num_workers 1 \
    # | tee ${log_dir}/train.log \
    # 2> ${log_dir}/train.err

deepspeed_args="--master_port=11111"
# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
CUDA_DEVICES=$CUDA_VISIBLE_DEVICES
#export CUDA_VISIBLE_DEVICES=0,1,6,7
deepspeed --include localhost:0,1,2,3,4,5,6,7 ${deepspeed_args} \
  kl_llama3_math_reward_ce.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_path autoregressive_reward_data_co8_gsm.json \
    --output_dir reward_llama_31_8b_autoregressive_ce8_gsm --overwrite_output_dir \
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
    --run_name reward_llama_31_8b_ce5 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 500000 \
    --save_only_model True \
    --dataloader_num_workers 1 \
    # | tee ${log_dir}/train.log \
    # 2> ${log_dir}/train.err

deepspeed_args="--master_port=11112"
# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
CUDA_DEVICES=$CUDA_VISIBLE_DEVICES
#export CUDA_VISIBLE_DEVICES=0,1,6,7
deepspeed --include localhost:0,1,2,3,4,5,6,7 ${deepspeed_args} \
  kl_llama3_math_reward_ce.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_path autoregressive_reward_data_co1_gsm.json \
    --output_dir reward_llama_31_8b_autoregressive_ce1_gsm --overwrite_output_dir \
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
    --run_name reward_llama_31_8b_ce5 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 500000 \
    --save_only_model True \
    --dataloader_num_workers 1 \
    # | tee ${log_dir}/train.log \
    # 2> ${log_dir}/train.err

deepspeed_args="--master_port=11113"
# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
CUDA_DEVICES=$CUDA_VISIBLE_DEVICES
#export CUDA_VISIBLE_DEVICES=0,1,6,7
deepspeed --include localhost:0,1,2,3,4,5,6,7 ${deepspeed_args} \
  kl_llama3_math_reward_ce.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_path autoregressive_reward_data_co0.1_gsm.json \
    --output_dir reward_llama_31_8b_autoregressive_ce0.1_gsm --overwrite_output_dir \
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
    --run_name reward_llama_31_8b_ce5 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 500000 \
    --save_only_model True \
    --dataloader_num_workers 1 \
    # | tee ${log_dir}/train.log \
    # 2> ${log_dir}/train.err

deepspeed_args="--master_port=11114"
# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
CUDA_DEVICES=$CUDA_VISIBLE_DEVICES
#export CUDA_VISIBLE_DEVICES=0,1,6,7
deepspeed --include localhost:0,1,2,3,4,5,6,7 ${deepspeed_args} \
  kl_llama3_math_reward_ce.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_path autoregressive_reward_data_co10_math.json \
    --output_dir reward_llama_31_8b_autoregressive_ce10_math --overwrite_output_dir \
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
    --run_name reward_llama_31_8b_ce5 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 500000 \
    --save_only_model True \
    --dataloader_num_workers 1 \
    # | tee ${log_dir}/train.log \
    # 2> ${log_dir}/train.err

deepspeed_args="--master_port=11115"
# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
CUDA_DEVICES=$CUDA_VISIBLE_DEVICES
#export CUDA_VISIBLE_DEVICES=0,1,6,7
deepspeed --include localhost:0,1,2,3,4,5,6,7 ${deepspeed_args} \
  kl_llama3_math_reward_ce.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_path autoregressive_reward_data_co8_math.json \
    --output_dir reward_llama_31_8b_autoregressive_ce8_math --overwrite_output_dir \
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
    --run_name reward_llama_31_8b_ce5 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 500000 \
    --save_only_model True \
    --dataloader_num_workers 1 \
    # | tee ${log_dir}/train.log \
    # 2> ${log_dir}/train.err

deepspeed_args="--master_port=11116"
# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
CUDA_DEVICES=$CUDA_VISIBLE_DEVICES
#export CUDA_VISIBLE_DEVICES=0,1,6,7
deepspeed --include localhost:0,1,2,3,4,5,6,7 ${deepspeed_args} \
  kl_llama3_math_reward_ce.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_path autoregressive_reward_data_co5_math.json \
    --output_dir reward_llama_31_8b_autoregressive_ce5_math --overwrite_output_dir \
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
    --run_name reward_llama_31_8b_ce5 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 500000 \
    --save_only_model True \
    --dataloader_num_workers 1 \
    # | tee ${log_dir}/train.log \
    # 2> ${log_dir}/train.err

deepspeed_args="--master_port=11117"
# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
CUDA_DEVICES=$CUDA_VISIBLE_DEVICES
#export CUDA_VISIBLE_DEVICES=0,1,6,7
deepspeed --include localhost:0,1,2,3,4,5,6,7 ${deepspeed_args} \
  kl_llama3_math_reward_ce.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_path autoregressive_reward_data_co2_math.json \
    --output_dir reward_llama_31_8b_autoregressive_ce2_math --overwrite_output_dir \
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
    --run_name reward_llama_31_8b_ce5 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 500000 \
    --save_only_model True \
    --dataloader_num_workers 1 \
    # | tee ${log_dir}/train.log \
    # 2> ${log_dir}/train.err

deepspeed_args="--master_port=11118"
# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
CUDA_DEVICES=$CUDA_VISIBLE_DEVICES
#export CUDA_VISIBLE_DEVICES=0,1,6,7
deepspeed --include localhost:0,1,2,3,4,5,6,7 ${deepspeed_args} \
  kl_llama3_math_reward_ce.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_path autoregressive_reward_data_co1_math.json \
    --output_dir reward_llama_31_8b_autoregressive_ce1_math --overwrite_output_dir \
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
    --run_name reward_llama_31_8b_ce5 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 500000 \
    --save_only_model True \
    --dataloader_num_workers 1 \
    # | tee ${log_dir}/train.log \
    # 2> ${log_dir}/train.err

deepspeed_args="--master_port=11119"
# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
CUDA_DEVICES=$CUDA_VISIBLE_DEVICES
#export CUDA_VISIBLE_DEVICES=0,1,6,7
deepspeed --include localhost:0,1,2,3,4,5,6,7 ${deepspeed_args} \
  kl_llama3_math_reward_ce.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_path autoregressive_reward_data_co0.1_math.json \
    --output_dir reward_llama_31_8b_autoregressive_ce0.1_math --overwrite_output_dir \
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
    --run_name reward_llama_31_8b_ce5 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 500000 \
    --save_only_model True \
    --dataloader_num_workers 1 \
    # | tee ${log_dir}/train.log \
    # 2> ${log_dir}/train.err