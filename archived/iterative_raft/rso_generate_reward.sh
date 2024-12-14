python rso_generate_reward.py  \
    --model_name_or_path $1 \
    --current_iter $2 \
    --batch_size 8192 \
    --batch_size_per_iter 20480 \
    --tensor_parallel_size 1 \
    --num_gpus $3 \
    --output_dir $4 \
    --sanity_check $5 \
    --local_rank $6
