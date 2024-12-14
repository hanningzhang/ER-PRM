# This is the entrance bash file to use reward model to score the sampled rationales, and select the best sample based on scores

python rso_reward.py \
    --dataset $1 \
    --output_dir $2 \
    --reward_name_or_path $3 \
    --num_gpus $4 \
    --local_rank $5
    