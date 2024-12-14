# This is the entrace bash file to use policy model to sample step wise ratioanles

python rso_generate.py  \
    --model_name_or_path $1 \
    --current_iter $2 \
    --dataset meta-math/MetaMathQA \
    --batch_size 25000 \
    --batch_size_per_iter 20000 \
    --tensor_parallel_size 1 \
    --num_gpus $3 \
    --output_dir $4 \
    --sanity_check $5 \
    --local_rank $6 \
    --random_seed $7 
