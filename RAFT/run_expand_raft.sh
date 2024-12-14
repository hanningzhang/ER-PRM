#!/bin/bash

# This is the entrance file for the RAFT Algorithm

# The total iteration of raft
num_iteration=3

base_dir="peiyi9979/mistral-7b-sft"
mkdir -p $base_dir
# You should edit the sft model dir accordingly
sft_model="meta-llama/Meta-Llama-3-8B-Instruct"
reward_model="reward_model_path"

sanity_check=0
num_gpus=4
gpu_list=0,1,2,3
IFS=',' read -r -a gpu_list_array <<< "$gpu_list"

random_seed=42

"""
Important Notes for using this script

1. Please config the above parameters as you need, especially the num_gpus and corresponding gpu_list

2. The numeric number at the end of each shell command is the local rank of GPU, make sure it has to be consecutive order
    ex: gpu_list=0,2,4,6, the relative local rank for these GPUs are still 0, 1, 2, 3

3. Comment out the line of shell command with GPUs that you don't use

4. When use the evaluate.sh script, use 1 GPU is sufficient for current project. More GPU doesn't necessarily faster.

5. This script expand the iterative for loop to different section, so that you can easily resart/continue the algorithm 
    is unexpected case happened, i.e. CUDA out of mem.

6. The sampling result from get_samples.sh will be saved into the current the infer_set dir of current model_dir

7. The get_reward.sh will score the data from infer_set and save the filtered result into the filterer_set of current model_dir

8. Evaluation result are saved into eval_result dir in the RAFT dir

9. The evaluation will load data from RAFT/data/test
"""

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


############### Iter 1

for ((j=0; j<num_gpus; j++))
do
    CUDA_VISIBLE_DEVICES=${gpu_list_array[$j]} bash ./get_samples.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} $j ${random_seed} &
done
wait

if [ $? -ne 0 ]; then echo "ERROR occur during sampling from the SFT Model"; exit 1; fi

for ((j=0; j<num_gpus; j++))
do
    bash ./get_rewards.sh ${model_dir}/infer_set ${model_dir}/filtered_set ${reward_model} ${num_gpus} $j &
done

wait

python merge_data.py --output_dir ${model_dir}/filtered_set
if [ $? -ne 0 ]; then echo "ERROR occur during calculating with the Reward Model"; exit 1; fi

bash ./sft.sh ${sft_model} $tmp_model_dir ${model_dir}/filtered_set $((1)) ${num_iteration} ${random_seed} ${gpu_list}
if [ $? -ne 0 ]; then echo "ERROR occur during Supervised Fine-tuning the Model"; exit 1; fi

CUDA_VISIBLE_DEVICES=${gpu_list_array[0]} bash ./evaluate.sh $tmp_model_dir 1 ${sft_model}
if [ $? -ne 0 ]; then echo "ERROR occur during the evaluation part"; exit 1; fi

old_model_dir=$tmp_model_dir 

############### Iter 2
i=2
model_dir="${base_dir}/model2"
mkdir -p $model_dir
mkdir -p ${model_dir}/infer_set
mkdir -p ${model_dir}/filtered_set

for ((j=0; j<num_gpus; j++))
do
    CUDA_VISIBLE_DEVICES=${gpu_list_array[$j]} bash ./get_samples.sh ${old_model_dir} $((i - 1)) ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} $j ${random_seed} &
done

wait

if [ $? -ne 0 ]; then echo "ERROR occur during sampling from the SFT Model"; exit 1; fi
for ((j=0; j<num_gpus; j++))
do
    bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${reward_model} ${num_gpus} $j &
done

wait

python merge_data.py --output_dir ${old_model_dir}/filtered_set
if [ $? -ne 0 ]; then echo "ERROR occur during calculating with the Reward Model"; exit 1; fi

bash ./sft.sh $old_model_dir $model_dir ${old_model_dir}/filtered_set $((i)) ${num_iteration} ${random_seed} ${gpu_list}
if [ $? -ne 0 ]; then echo "ERROR occur during Supervised Fine-tuning the Model"; exit 1; fi

CUDA_VISIBLE_DEVICES=${gpu_list_array[0]} bash ./evaluate.sh $model_dir 1 ${sft_model}
if [ $? -ne 0 ]; then echo "ERROR occur during the evaluation part"; exit 1; fi

############### Iter 3

i=3
old_model_dir=$model_dir

model_dir="${base_dir}/model${i}"
mkdir -p $model_dir
mkdir -p ${model_dir}/infer_set
mkdir -p ${model_dir}/filtered_set

for ((j=0; j<num_gpus; j++))
do
    CUDA_VISIBLE_DEVICES=${gpu_list_array[$j]} bash ./get_samples.sh ${old_model_dir} $((i - 1)) ${num_gpus} ${old_model_dir}/infer_set ${sanity_check} $j ${random_seed} &
done

wait

if [ $? -ne 0 ]; then echo "ERROR occur during sampling from the SFT Model"; exit 1; fi
for ((j=0; j<num_gpus; j++))
do
    bash ./get_rewards.sh ${old_model_dir}/infer_set ${old_model_dir}/filtered_set ${reward_model} ${num_gpus} $j &
done

wait

python merge_data.py --output_dir ${old_model_dir}/filtered_set
if [ $? -ne 0 ]; then echo "ERROR occur during calculating with the Reward Model"; exit 1; fi

bash ./sft.sh $old_model_dir $model_dir ${old_model_dir}/filtered_set $((i)) ${num_iteration} ${random_seed} ${gpu_list}
if [ $? -ne 0 ]; then echo "ERROR occur during Supervised Fine-tuning the Model"; exit 1; fi

CUDA_VISIBLE_DEVICES=${gpu_list_array[0]} bash ./evaluate.sh $model_dir 1 ${sft_model}
if [ $? -ne 0 ]; then echo "ERROR occur during the evaluation part"; exit 1; fi

old_model_dir=$model_dir
