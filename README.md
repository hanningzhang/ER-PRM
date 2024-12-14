# Process Reward Model Training and Reinforcement Learning from Human Feedback

We borrow the environment setting from `https://github.com/RLHFlow/Online-RLHF`
## Environment Setup

- clone and create a conda environment
```
conda create -n math python=3.10.9 --yes
conda activate math

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout d17fd7cd3b71c6a7bf7af34d8dc73135bb7ea8e9
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .
pip install https://github.com/vllm-project/vllm/releases/download/v0.4.0/vllm-0.4.0-cp310-cp310-manylinux1_x86_64.whl
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install accelerate==0.27.2
pip install datasets deepspeed wandb stanza fraction jsonlines spacy
pip install --upgrade trl
conda install ninja

pip install wandb

export SSL_CERT_DIR='/etc/ssl/certs'
export REQUESTS_CA_BUNDLE='/etc/ssl/certs/ca-certificates.crt'

```
- track the training with wandb
```
wandb login
huggingface-cli login
```
# Process Reward Model (PRM) Training

## Dataset Generation
Go to `\MCTS` folder

Here we provide an example of generating completions for `deepseek-math-7b-rl` model using datasets from `HanningZhang/deepseek-gsm-new` and `HanningZhang/deepseek-math-new` on Huggingface.

**You should change your own model and dataset for the generation.**

run
```
python -m spacy download en_core_web_sm
bash run_gsm_split0.sh
bash run_math_split0.sh
```
This will generate multiple completions for each step.

Then 
`python merge_data.py` to merge all the pieces of data.

Then
```
Python calculate_score.py --coefficient 1
```
to label the data with customized coefficients.

Run `python convert_autoregressive.py` to convert into auto-regressive format.


## Dataset Preparation
Please prepare a JSON file with the following format:
```
[
    {
        "text": "Janet pays $40/hour for 3 hours per week of clarinet lessons.."    
        "value": [0.8, 0.6, 0.3 ...]
    },
    {
        "text": "Val cuts a single watermelon into 40 slices.."    
        "value": [0.5, 0.7, 0.2 ...]
    }
]
```
Here is a full example of a text
```
Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year? Step 1: Janet spends 3 hours + 5 hours = <<3+5=8>>8 hours per week on music lessons. ки\nStep 2: She spends 40 * 3 = <<40*3=120>>120 on clarinet lessons per week. ки\nStep 3: She spends 28 * 5 = <<28*5=140>>140 on piano lessons per week. ки\nStep 4: Janet spends 120 + 140 = <<120+140=260>>260 on music lessons per week. ки\nStep 5: She spends 260 * 52 = <<260*52=13520>>13520 on music lessons in a year. The answer is: 13520 ки
```
We use `ки\n` to separate each step.


Here is a full example of a value
```
[1.0, 0.8, 0.6, 0.7, 1.0]
```
The length of the list must be the same as the number of `ки`, as we want to train the reward model to predict the process reward we specify.

## Training Code
Go to `\prm` folder.

Please run the following bash file
```
bash kl_math_reward_ce.sh
```
You can specify the hyper-parameters within this file (`data_path`, `model_name`, `learning_rate`, `batch_size`...)

## Evaluation
Go to `evaluation` folder.

The usage of this model is very similar to Math-Shpeherd.

When the reward model encounters `ки`, it will predict the score for current step.

The calculation is the result of the probability of `+` after a softmax of `+` and `-` logits.

Below is an example code:
```
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

good_token = ' +'
bad_token = ' -'
step_tag = 'ки'

tokenizer = AutoTokenizer.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm')
plus_id = tokenizer.encode(' +')[-1]
minus_id = tokenizer.encode(' -')[-1]
candidate_tokens = [plus_id, minus_id]
step_tag_id = tokenizer.encode(f"{step_tag}")[-1] # 12902
model = AutoModelForCausalLM.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm').eval()

question = """Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
output1 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки""" # 18 is right
output2 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки""" # 17 is wrong
output1 = output1.replace("ки\n","ки")
output2 = output2.replace("ки\n","ки")
for output in [output1, output2]:
    input_for_prm = f"{question} {output}"
    input_id = torch.tensor([tokenizer.encode(input_for_prm)])

    with torch.no_grad():
        logits = model(input_id).logits[:,:,candidate_tokens]
        scores = logits.softmax(dim=-1)[:,:,0] 
        step_scores = scores[input_id == step_tag_id]
        print(step_scores)
```

# RAFT
Go to `\RAFT` folder.

We use RAFT algorithm to enhance RLHF with our proposed ER-PRM

## Overall Flow of Pipeline
Notice: This script implement multi-process running by parallel running multiple bash script, therefore will generate multiple files. All relevant code are placed in the RAFT directory.

1. First open the run_expand_raft.sh and config some important parameters (Details see next section).

2. This script contains several core scripts for different tasks, which are "get_samples.sh", "get_reward.sh", "merge_data.py", "sft.sh", "evaluate.sh".

3. The entire flow starts with get_samples.sh, it will use the policy model to sample step by step rationales to the given dataset. The dataset will pull from hugging face with the given name. The sampled rationales will be output to the infer_set dir of current iteration dir.

4. Second step is get_reward.sh, it will use the reward model specified to score the sampled rationales and select the best one for later SFT. It's input comes from the infer_set of current dir, and output the selected rationales to filter_set of current dir"

5. Third step is the merge_data.py, it simply merges the output files from different GPU to a single file for later SFT.

6. Fourth step is the sft.sh, it will take the selected rationales from filter_set of current iteration dir as training data. The fine-tuned model after training will be saved into next iteration's work dir. i.e. if current work dir is model0, the fine-tuned model will be saved to model1.

7. Fifth step is evaluate.sh, it will test the fine-tuned model on gsm8k and math datasets.

8. The algorithm will repeat the above steps in new iterations until reach the specified number of iteration.

## Important Notes for using the restart_raft.sh script
All relevant code are placed in the RAFT directory.

Use run_expand_raft.sh, and config some important parameters

1. Please config the parameters as you need, especially the num_iteration, num_gpus and corresponding gpu_list

2. The num_iteration matches with the number of RAFT block in the bash script, default is 3 iteration, you can comment out or add new block for less or more iterations.

2. When use the evaluate.sh script, use 1 GPU is sufficient for current project. More GPU doesn't necessarily faster.

3. This script expand the iterative for loop to different section, so that you can easily resart/continue the algorithm in case unexpected case happened, i.e. CUDA out of mem.

4. Evaluation result are saved into eval_result dir in the RAFT dir

5. The evaluation will load test data from data/test

6. The sanity_check parameter represents a test run with small amount of data if set to 1, default to 0

## Example of configuration of restart_raft.sh
```bash
# specify the base_dir as where to store the models during iterative process, 
num_iteration=3

base_dir="iterative_llama_pool_prm"
mkdir -p $base_dir
# You should edit the sft model dir accordingly
sft_model="PATH_TO_POLICY_MODEL"
reward_model="PATH_TO_REWRAD_MODEL"

# set to 1 if want to verify the pipeline with small amount of data
sanity_check=0
# specify the number and which GPUs to use
num_gpus=4
gpu_list=0,1,2,3

# set random seed
random_seed=42
```


## Troubleshooting
Q: `ModuleNotFoundError: No module named 'packaging'`
A: `pip install packaging`

Q: duirng installation encoutered ModuleNotFoundError: No module named 'torch'.
A: I found that `pip install -r requirements.txt` could install `pytorch` for me. Therefore, please make sure you have installed pytorch. If not, please install it by following the instructions on the official website: https://pytorch.org/get-started/locally/

Q: `not success in downloading stanza. Retrying....`
A: may be caused by `out of memory`, check the GPU usage. 

Q: `RuntimeError: Failed to import transformers.models.llama.modeling_llama because of the following error (look up to see its traceback): /home/sdiaoaa/anaconda3/envs/math/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZNK3c1017SymbolicShapeMeta18init_is_contiguousEv`
A: `pip uninstall flash-attn`
