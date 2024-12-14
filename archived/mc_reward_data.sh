python mc_reward_data.py \
    --completion_model_name_or_path peiyi9979/mistral-7b-sft \
    --dataset_path peiyi9979/Math-Shepherd \
    --output_dir smooth_reward_data \
    --tensor_parallel_size 1 \
    --num_gpus 8 \
    --local_rank $1 \
    --sampling_num 16