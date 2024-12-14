#!/bin/bash

# The total iteration of raft
num_iteration=3


base_dir="iterative_dpo_instruct_2e-7_prm"
mkdir -p $base_dir
# You should edit the sft model dir accordingly
sft_model="meta-llama/Meta-Llama-3-8B-Instruct"
reward_model="peiyi9979/math-shepherd-mistral-7b-prm"

sanity_check=0
num_gpus=8

random_seed=42

x=0
y=1
model_dir="${base_dir}/model${x}"
mkdir -p ${model_dir}
tmp_model_dir="${base_dir}/model${y}"

mkdir -p $tmp_model_dir
mkdir -p ${model_dir}/infer_set
mkdir -p ${model_dir}/filtered_set
mkdir -p ${tmp_model_dir}/infer_set
mkdir -p ${tmp_model_dir}/filtered_set
mkdir -p ${tmp_model_dir}/infer_set_for_reward
mkdir -p ${tmp_model_dir}/reward_data
mkdir -p ${tmp_model_dir}/reward_model

CUDA_VISIBLE_DEVICES=0 bash ./get_samples_instruct.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 0 ${random_seed} &
CUDA_VISIBLE_DEVICES=1 bash ./get_samples_instruct.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 1 ${random_seed} &
CUDA_VISIBLE_DEVICES=2 bash ./get_samples_instruct.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 2 ${random_seed} &
CUDA_VISIBLE_DEVICES=3 bash ./get_samples_instruct.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 3 ${random_seed} &
CUDA_VISIBLE_DEVICES=4 bash ./get_samples_instruct.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 4 ${random_seed} &
CUDA_VISIBLE_DEVICES=5 bash ./get_samples_instruct.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 5 ${random_seed} &
CUDA_VISIBLE_DEVICES=6 bash ./get_samples_instruct.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 6 ${random_seed} &
CUDA_VISIBLE_DEVICES=7 bash ./get_samples_instruct.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 7 ${random_seed} &

wait
if [ $? -ne 0 ]; then echo "ERROR occur during sampling from the SFT Model"; exit 1; fi

bash ./dpo_reward.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${reward_model} ${num_gpus} 0 &
bash ./dpo_reward.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${reward_model} ${num_gpus} 1 &
bash ./dpo_reward.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${reward_model} ${num_gpus} 2 &
bash ./dpo_reward.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${reward_model} ${num_gpus} 3 &
bash ./dpo_reward.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${reward_model} ${num_gpus} 4 &
bash ./dpo_reward.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${reward_model} ${num_gpus} 5 &
bash ./dpo_reward.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${reward_model} ${num_gpus} 6 &
bash ./dpo_reward.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${reward_model} ${num_gpus} 7 &

wait
python merge_data.py --output_dir ${model_dir}/filtered_set
if [ $? -ne 0 ]; then echo "ERROR occur during calculating with the Reward Model"; exit 1; fi
# python filter_gsm.py --output_dir ${model_dir}/filtered_set
bash ./run_dpo.sh ${sft_model} $tmp_model_dir ${model_dir}/filtered_set $((1)) ${num_iteration} ${random_seed} ${sft_model}
if [ $? -ne 0 ]; then echo "ERROR occur during Supervised Fine-tuning the Model"; exit 1; fi
#bash ./evaluate.sh $tmp_model_dir ${num_gpus} ${sft_model}
CUDA_VISIBLE_DEVICES=0 bash ./evaluate.sh $tmp_model_dir $((1)) ${sft_model}
if [ $? -ne 0 ]; then echo "ERROR occur during the evaluation part"; exit 1; fi
old_model_dir=$tmp_model_dir 
