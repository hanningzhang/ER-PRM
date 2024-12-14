python mc_process.py \
    --completion_model_name_or_path $1 \
    --dataset_path $2 \
    --output_dir $3 \
    --tensor_parallel_size 1 \
    --num_gpus $4 \
    --local_rank $5 \
    --reference_dataset meta-math/MetaMathQA \
    --max_number 4096