import pandas as pd
import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default = 'MetaMath-2b',help="model name could be MetaMath-2b, MetaMath-7b...")
    return parser.parse_args()

args = parse_args()
model_name = args.model

file_list = os.listdir("eval_result")
gsm8k_list = []
math_list = []
for file in file_list:
    if "gsm" in file.lower()[:3] and model_name.lower() in file.lower():
        gsm8k_list.append(file)
    if "math" in file.lower()[:4] and model_name.lower() in file.lower():
        math_list.append(file)
        
assert len(gsm8k_list) == len(math_list)

full_result = []
count = 1
for gsm_instance, math_instance in zip(gsm8k_list,math_list):
    with open(f"eval_result/{gsm_instance}", "r") as f:
        gsm8k_dict = (json.load(f))
    with open(f"eval_result/{math_instance}", "r") as f:
        math_dict = (json.load(f))
    full_result.append({"iter": count, "gsm8k": gsm8k_dict["gsm8k"], "math": math_dict["math"]})
    count += 1
# for i in range(10):
#     try:
#         with open(F"eval_result/gsm8k_iterative_rft_model{i+1}.json", "r") as f:
#             gsm8k_dict = (json.load(f))
#         with open(F"eval_result/math_iterative_rft_model{i+1}.json", "r") as f:
#             math_dict = (json.load(f))
#         full_result.append({"iter": i+1, "gsm8k": gsm8k_dict["gsm8k"], "math": math_dict["math"]})
#     except:
#         continue
print(pd.DataFrame(full_result))
