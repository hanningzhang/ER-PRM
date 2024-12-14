python mc_reward_data_gsm.py \
    --completion_model_name_or_path deepseek-ai/deepseek-math-7b-rl \
    --dataset_path HanningZhang/deepseek-gsm-new \
    --output_dir smooth_reward_data \
    --tensor_parallel_size 1 \
    --num_gpus 8 \
    --local_rank $1 \
    --sampling_num 16 \
    --split $2 
