#!/bin/bash

# The total iteration of raft
num_iteration=3

base_dir="metamath_7b_iterative"
mkdir -p $base_dir
# You should edit the sft model dir accordingly
sft_model="HanningZhang/MetaMath-7b"
reward_model="HanningZhang/MathReward-7b"

sanity_check=0
num_gpus=8

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

CUDA_VISIBLE_DEVICES=0 bash ./get_samples.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 0 &
CUDA_VISIBLE_DEVICES=1 bash ./get_samples.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 1 &
CUDA_VISIBLE_DEVICES=2 bash ./get_samples.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 2 &
CUDA_VISIBLE_DEVICES=3 bash ./get_samples.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 3 &
CUDA_VISIBLE_DEVICES=4 bash ./get_samples.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 4 &
CUDA_VISIBLE_DEVICES=5 bash ./get_samples.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 5 &
CUDA_VISIBLE_DEVICES=6 bash ./get_samples.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 6 &
CUDA_VISIBLE_DEVICES=7 bash ./get_samples.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 7 &

wait
if [ $? -ne 0 ]; then echo "ERROR occur during sampling from the SFT Model"; exit 1; fi
bash ./get_rewards.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${reward_model} ${num_gpus} 0 &
bash ./get_rewards.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${reward_model} ${num_gpus} 1 &
bash ./get_rewards.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${reward_model} ${num_gpus} 2 &
bash ./get_rewards.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${reward_model} ${num_gpus} 3 &
bash ./get_rewards.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${reward_model} ${num_gpus} 4 &
bash ./get_rewards.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${reward_model} ${num_gpus} 5 &
bash ./get_rewards.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${reward_model} ${num_gpus} 6 &
bash ./get_rewards.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${reward_model} ${num_gpus} 7 &

wait
python merge_data.py --output_dir ${model_dir}/filtered_set
if [ $? -ne 0 ]; then echo "ERROR occur during calculating with the Reward Model"; exit 1; fi
bash ./sft.sh ${sft_model} $tmp_model_dir ${model_dir}/filtered_set $((1)) ${num_iteration}
if [ $? -ne 0 ]; then echo "ERROR occur during Supervised Fine-tuning the Model"; exit 1; fi
bash ./evaluate.sh $tmp_model_dir ${num_gpus} ${sft_model}
if [ $? -ne 0 ]; then echo "ERROR occur during the evaluation part"; exit 1; fi
python evaluate_reward_model.py --reward_name_or_path ${reward_model} --num_gpus ${num_gpus}

CUDA_VISIBLE_DEVICES=0 bash ./rso_generate_reward.sh ${tmp_model_dir} 0 ${num_gpus} ${tmp_model_dir}/infer_set_for_reward ${sanity_check} 0 &
CUDA_VISIBLE_DEVICES=1 bash ./rso_generate_reward.sh ${tmp_model_dir} 0 ${num_gpus} ${tmp_model_dir}/infer_set_for_reward ${sanity_check} 1 &
CUDA_VISIBLE_DEVICES=2 bash ./rso_generate_reward.sh ${tmp_model_dir} 0 ${num_gpus} ${tmp_model_dir}/infer_set_for_reward ${sanity_check} 2 &
CUDA_VISIBLE_DEVICES=3 bash ./rso_generate_reward.sh ${tmp_model_dir} 0 ${num_gpus} ${tmp_model_dir}/infer_set_for_reward ${sanity_check} 3 &
CUDA_VISIBLE_DEVICES=4 bash ./rso_generate_reward.sh ${tmp_model_dir} 0 ${num_gpus} ${tmp_model_dir}/infer_set_for_reward ${sanity_check} 4 &
CUDA_VISIBLE_DEVICES=5 bash ./rso_generate_reward.sh ${tmp_model_dir} 0 ${num_gpus} ${tmp_model_dir}/infer_set_for_reward ${sanity_check} 5 &
CUDA_VISIBLE_DEVICES=6 bash ./rso_generate_reward.sh ${tmp_model_dir} 0 ${num_gpus} ${tmp_model_dir}/infer_set_for_reward ${sanity_check} 6 &
CUDA_VISIBLE_DEVICES=7 bash ./rso_generate_reward.sh ${tmp_model_dir} 0 ${num_gpus} ${tmp_model_dir}/infer_set_for_reward ${sanity_check} 7 &

wait
if [ $? -ne 0 ]; then echo "ERROR occur during sampling from the SFT Model"; exit 1; fi

CUDA_VISIBLE_DEVICES=0 bash ./mc_process.sh ${sft_model} ${tmp_model_dir}/infer_set_for_reward ${tmp_model_dir}/reward_data ${num_gpus} 0 &
CUDA_VISIBLE_DEVICES=1 bash ./mc_process.sh ${sft_model} ${tmp_model_dir}/infer_set_for_reward ${tmp_model_dir}/reward_data ${num_gpus} 1 &
CUDA_VISIBLE_DEVICES=2 bash ./mc_process.sh ${sft_model} ${tmp_model_dir}/infer_set_for_reward ${tmp_model_dir}/reward_data ${num_gpus} 2 &
CUDA_VISIBLE_DEVICES=3 bash ./mc_process.sh ${sft_model} ${tmp_model_dir}/infer_set_for_reward ${tmp_model_dir}/reward_data ${num_gpus} 3 &
CUDA_VISIBLE_DEVICES=4 bash ./mc_process.sh ${sft_model} ${tmp_model_dir}/infer_set_for_reward ${tmp_model_dir}/reward_data ${num_gpus} 4 &
CUDA_VISIBLE_DEVICES=5 bash ./mc_process.sh ${sft_model} ${tmp_model_dir}/infer_set_for_reward ${tmp_model_dir}/reward_data ${num_gpus} 5 &
CUDA_VISIBLE_DEVICES=6 bash ./mc_process.sh ${sft_model} ${tmp_model_dir}/infer_set_for_reward ${tmp_model_dir}/reward_data ${num_gpus} 6 &
CUDA_VISIBLE_DEVICES=7 bash ./mc_process.sh ${sft_model} ${tmp_model_dir}/infer_set_for_reward ${tmp_model_dir}/reward_data ${num_gpus} 7 &

wait
python merge_data.py --output_dir ${tmp_model_dir}/reward_data
if [ $? -ne 0 ]; then echo "ERROR occur during Monto Carlo labeling"; exit 1; fi
bash ./train_reward.sh ${reward_model} ${tmp_model_dir}/reward_data ${tmp_model_dir}/reward_model $((1)) ${num_iteration}
if [ $? -ne 0 ]; then echo "ERROR occur during sampling from the fine-tuning the reward model"; exit 1; fi

old_model_dir=$tmp_model_dir 

for (( i=2; i<$num_iteration; i++ )); do
  model_dir="${base_dir}/model${i}"
  mkdir -p $model_dir
  mkdir -p ${model_dir}/infer_set
  mkdir -p ${model_dir}/filtered_set
  mkdir -p ${model_dir}/infer_set_for_reward
  mkdir -p ${model_dir}/reward_data
  mkdir -p ${model_dir}/reward_model
  
  CUDA_VISIBLE_DEVICES=0 bash ./get_samples.sh ${old_model_dir} 0 ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} 0 &
  CUDA_VISIBLE_DEVICES=1 bash ./get_samples.sh ${old_model_dir} 0 ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} 1 &
  CUDA_VISIBLE_DEVICES=2 bash ./get_samples.sh ${old_model_dir} 0 ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} 2 &
  CUDA_VISIBLE_DEVICES=3 bash ./get_samples.sh ${old_model_dir} 0 ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} 3 &
  CUDA_VISIBLE_DEVICES=4 bash ./get_samples.sh ${old_model_dir} 0 ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} 4 &
  CUDA_VISIBLE_DEVICES=5 bash ./get_samples.sh ${old_model_dir} 0 ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} 5 &
  CUDA_VISIBLE_DEVICES=6 bash ./get_samples.sh ${old_model_dir} 0 ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} 6 &
  CUDA_VISIBLE_DEVICES=7 bash ./get_samples.sh ${old_model_dir} 0 ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} 7 &

  wait
  if [ $? -ne 0 ]; then echo "ERROR occur during sampling from the SFT Model"; exit 1; fi
  bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${old_model_dir}/reward_model ${num_gpus} 0 &
  bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${old_model_dir}/reward_model ${num_gpus} 1 &
  bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${old_model_dir}/reward_model ${num_gpus} 2 &
  bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${old_model_dir}/reward_model ${num_gpus} 3 &
  bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${old_model_dir}/reward_model ${num_gpus} 4 &
  bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${old_model_dir}/reward_model ${num_gpus} 5 &
  bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${old_model_dir}/reward_model ${num_gpus} 6 &
  bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${old_model_dir}/reward_model ${num_gpus} 7 &

  wait
  python merge_data.py --output_dir ${old_model_dir}/filtered_set
  if [ $? -ne 0 ]; then echo "ERROR occur during calculating with the Reward Model"; exit 1; fi
  bash ./sft.sh $old_model_dir $model_dir ${old_model_dir}/filtered_set $((i)) ${num_iteration}
  if [ $? -ne 0 ]; then echo "ERROR occur during Supervised Fine-tuning the Model"; exit 1; fi
  bash ./evaluate.sh $model_dir ${num_gpus} ${sft_model}
  if [ $? -ne 0 ]; then echo "ERROR occur during the evaluation part"; exit 1; fi
  python evaluate_reward_model.py --reward_name_or_path ${old_model_dir}/reward_model --num_gpus ${num_gpus}

  CUDA_VISIBLE_DEVICES=0 bash ./rso_generate_reward.sh ${model_dir} $((i - 1)) ${num_gpus} ${model_dir}/infer_set_for_reward ${sanity_check} 0 &
  CUDA_VISIBLE_DEVICES=1 bash ./rso_generate_reward.sh ${model_dir} $((i - 1)) ${num_gpus} ${model_dir}/infer_set_for_reward ${sanity_check} 1 &
  CUDA_VISIBLE_DEVICES=2 bash ./rso_generate_reward.sh ${model_dir} $((i - 1)) ${num_gpus} ${model_dir}/infer_set_for_reward ${sanity_check} 2 &
  CUDA_VISIBLE_DEVICES=3 bash ./rso_generate_reward.sh ${model_dir} $((i - 1)) ${num_gpus} ${model_dir}/infer_set_for_reward ${sanity_check} 3 &
  CUDA_VISIBLE_DEVICES=4 bash ./rso_generate_reward.sh ${model_dir} $((i - 1)) ${num_gpus} ${model_dir}/infer_set_for_reward ${sanity_check} 4 &
  CUDA_VISIBLE_DEVICES=5 bash ./rso_generate_reward.sh ${model_dir} $((i - 1)) ${num_gpus} ${model_dir}/infer_set_for_reward ${sanity_check} 5 &
  CUDA_VISIBLE_DEVICES=6 bash ./rso_generate_reward.sh ${model_dir} $((i - 1)) ${num_gpus} ${model_dir}/infer_set_for_reward ${sanity_check} 6 &
  CUDA_VISIBLE_DEVICES=7 bash ./rso_generate_reward.sh ${model_dir} $((i - 1)) ${num_gpus} ${model_dir}/infer_set_for_reward ${sanity_check} 7 &

  wait
  if [ $? -ne 0 ]; then echo "ERROR occur during sampling from the SFT Model"; exit 1; fi

  CUDA_VISIBLE_DEVICES=0 bash ./mc_process.sh ${sft_model} ${model_dir}/infer_set_for_reward ${model_dir}/reward_data ${num_gpus} 0 &
  CUDA_VISIBLE_DEVICES=1 bash ./mc_process.sh ${sft_model} ${model_dir}/infer_set_for_reward ${model_dir}/reward_data ${num_gpus} 1 &
  CUDA_VISIBLE_DEVICES=2 bash ./mc_process.sh ${sft_model} ${model_dir}/infer_set_for_reward ${model_dir}/reward_data ${num_gpus} 2 &
  CUDA_VISIBLE_DEVICES=3 bash ./mc_process.sh ${sft_model} ${model_dir}/infer_set_for_reward ${model_dir}/reward_data ${num_gpus} 3 &
  CUDA_VISIBLE_DEVICES=4 bash ./mc_process.sh ${sft_model} ${model_dir}/infer_set_for_reward ${model_dir}/reward_data ${num_gpus} 4 &
  CUDA_VISIBLE_DEVICES=5 bash ./mc_process.sh ${sft_model} ${model_dir}/infer_set_for_reward ${model_dir}/reward_data ${num_gpus} 5 &
  CUDA_VISIBLE_DEVICES=6 bash ./mc_process.sh ${sft_model} ${model_dir}/infer_set_for_reward ${model_dir}/reward_data ${num_gpus} 6 &
  CUDA_VISIBLE_DEVICES=7 bash ./mc_process.sh ${sft_model} ${model_dir}/infer_set_for_reward ${model_dir}/reward_data ${num_gpus} 7 &

  wait
  python merge_data.py --output_dir ${tmp_model_dir}/reward_data
  if [ $? -ne 0 ]; then echo "ERROR occur during Monto Carlo labeling"; exit 1; fi
  bash ./train_reward.sh ${old_model_dir}/reward_model ${model_dir}/reward_data ${model_dir}/reward_model $((i)) ${num_iteration}
  if [ $? -ne 0 ]; then echo "ERROR occur during sampling from the fine-tuning the reward model"; exit 1; fi

  old_model_dir=$model_dir
done

model_dir="${base_dir}/model${num_iteration}"
mkdir -p $model_dir
mkdir -p ${model_dir}/infer_set
mkdir -p ${model_dir}/filtered_set
mkdir -p ${model_dir}/infer_set_for_reward
mkdir -p ${model_dir}/reward_data
mkdir -p ${model_dir}/reward_model

CUDA_VISIBLE_DEVICES=0 bash ./get_samples.sh ${old_model_dir} $((num_iteration - 1)) ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} 0 &
CUDA_VISIBLE_DEVICES=1 bash ./get_samples.sh ${old_model_dir} $((num_iteration - 1)) ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} 1 &
CUDA_VISIBLE_DEVICES=2 bash ./get_samples.sh ${old_model_dir} $((num_iteration - 1)) ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} 2 &
CUDA_VISIBLE_DEVICES=3 bash ./get_samples.sh ${old_model_dir} $((num_iteration - 1)) ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} 3 &
CUDA_VISIBLE_DEVICES=4 bash ./get_samples.sh ${old_model_dir} $((num_iteration - 1)) ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} 4 &
CUDA_VISIBLE_DEVICES=5 bash ./get_samples.sh ${old_model_dir} $((num_iteration - 1)) ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} 5 &
CUDA_VISIBLE_DEVICES=6 bash ./get_samples.sh ${old_model_dir} $((num_iteration - 1)) ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} 6 &
CUDA_VISIBLE_DEVICES=7 bash ./get_samples.sh ${old_model_dir} $((num_iteration - 1)) ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} 7 &

wait
if [ $? -ne 0 ]; then echo "ERROR occur during sampling from the SFT Model"; exit 1; fi
bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${old_model_dir}/reward_model ${num_gpus} 0 &
bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${old_model_dir}/reward_model ${num_gpus} 1 &
bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${old_model_dir}/reward_model ${num_gpus} 2 &
bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${old_model_dir}/reward_model ${num_gpus} 3 &
bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${old_model_dir}/reward_model ${num_gpus} 4 &
bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${old_model_dir}/reward_model ${num_gpus} 5 &
bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${old_model_dir}/reward_model ${num_gpus} 6 &
bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${old_model_dir}/reward_model ${num_gpus} 7 &

wait
python merge_data.py --output_dir ${old_model_dir}/filtered_set
if [ $? -ne 0 ]; then echo "ERROR occur during calculating with the Reward Model"; exit 1; fi
bash ./sft.sh ${old_model_dir} $model_dir ${old_model_dir}/filtered_set ${num_iteration} ${num_iteration}
if [ $? -ne 0 ]; then echo "ERROR occur during Supervised Fine-tuning the Model"; exit 1; fi
bash ./evaluate.sh $model_dir ${num_gpus}
if [ $? -ne 0 ]; then echo "ERROR occur during the evaluation part"; exit 1; fi
python evaluate_reward_model.py --reward_name_or_path ${old_model_dir}/reward_model --num_gpus ${num_gpus}
