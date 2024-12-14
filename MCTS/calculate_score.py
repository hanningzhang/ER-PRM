import json
import re
from tqdm import tqdm
import stanza
from dataclasses import dataclass, field
from typing import Optional
from vllm import LLM, SamplingParams
from util import is_number, extract_answer_number
#from eval_math import remove_boxed, process_results
import util
import time
import os
import math
import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, pipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coefficient", type=int, default=1)  # model path
    return parser.parse_args()

args = parse_args()

def process_list(answer_list):
    new_list = [answer_list[0]]
    old = answer_list[0]['prompt']
    for sample in answer_list[1:-1]:
        if len(sample['prompt']) - len(old) > 30:
            new_list.append(sample)
            old = sample['prompt']
    new_list.append(answer_list[-1])
        
    return new_list

def remove_left_right(s):
    left = "\\left{"
    idx = s.find(left)
    if idx >= 0:
        text = s[idx+len(left):]
        right_idx = text.find("\\right")
        return text[:right_idx]
    else:
        return s
    
def check_and_remove_box(s):
    if "$\\boxed{" in s:
        start_idx = s.find("$\\boxed{")
        end_idx = s.find("}$")
        return s[start_idx+len("$\\boxed{"):end_idx]
    elif "$" in s:
        start_idx = s.find("$")
        s = s[start_idx+len("$"):]
        end_idx = s.find("$")
        return s[:end_idx]
    else:
        return s
    
def remove_boxed(s):
    left = "\\boxed{"
    idx = s.find(left)
    if idx == 0:
        #s = s[idx:]
        #right_idx = s.rfind("}")
        return s[idx+len(left):-1]
    else:
        return s
    
def remove_dollar(s):
    left = "$"
    if s[:len(left)] == left:
        s = s[1:]
        idx = s.find("$")
        return s[:idx]
    else:
        return s
    
def remove_final_dollar(s):
    left = "$"
    idx = s.find(left)
    if idx >= 0 and s.count("$") == 2:
        s = s[len(left)+idx:]
        right_idx = s.rfind("$")
        return s[:right_idx]
    else:
        return s

def remove_text_box(s):
    left = "\\text{"
    idx = s.find(left)
    if idx == 0:
        return s[idx+len(left):-1]
    else:
        return s
    
def remove_square_box(s):
    left = "\\["
    idx = s.find(left)
    if idx >= 0:
        right_idx = s.find("\\]")
        return s[idx+len(left):right_idx]
    else:
        return s
 
def remove_circle_box(s):
    if len(s) != 3:
        return s 
    left = "("
    idx = s.find(left)
    if idx >= 0:
        return s[len(left):-1]
    else:
        return s

def remove_mbox(s):
    left = "\\mbox{ "
    idx = s.find(left)
    if idx >= 0:
        return s[:idx].strip()
    else:
        return s    
    
def remove_text_box(s):
    left = "\\text{"
    idx = s.find(left)
    if idx >= 0:
        return s[len(left):-1]
    else:
        return s
    
def check_math_answer(content,ground_truth):
    #print(content)
    split_ans = content.split('The answer is')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        if len(ans) == 0:
            return False
        if ans[0] == ":":
            ans = ans[1:]
        ans = ans.strip()
        #extract_ans_temp = ans.split('.\n')[0]
        #extract_ans_temp = ans.split('ки')[0]
        extract_ans_temp = ans
        #extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        #print(extract_ans)
        extract_ans = remove_dollar(extract_ans)
        extract_ans = remove_dollar(extract_ans)
        extract_ans = remove_boxed(extract_ans)
        #print(extract_ans)
        extract_ans = remove_text_box(extract_ans)
        extract_ans = remove_square_box(extract_ans)
        extract_ans = remove_circle_box(extract_ans)
        extract_ans = remove_final_dollar(extract_ans)
        extract_ans = extract_ans.strip()
        #print(extract_ans)
        gt_ans = remove_boxed(util.last_boxed_only_string(ground_truth))
        if util.is_equiv(extract_ans, gt_ans):
            return True
        else:
            return False
    else:
        return False
    
def check_answer(sample,content):

    if "GSM" in sample['task']:
        temp_ans = sample['ground_truth'].split('#### ')[1]
        temp_ans = int(temp_ans.replace(',', ''))
        #print(float(extract_answer_number(sample['ground_truth'])))
        if not extract_answer_number(content):
            return 0
        if float(extract_answer_number(content)) == float(temp_ans):
            label = 1
        else:
            label = 0
    else:
        if check_math_answer(content,sample['ground_truth']):
            label = 1
        else:
            label = 0
            
    return label

def normalize_score(label_list,coefficient=1.0):
    sum = 0
    coefficient = args.coefficient
    num_0 = label_list.count(0)
    num_1 = label_list.count(1)
    sum += num_0/len(label_list) * math.exp((1)*coefficient*0)
    sum += num_1/len(label_list) * math.exp((1)*coefficient*1)
    log_sum = math.log(sum)
    
    return math.fabs((1)/coefficient * log_sum)
    # return math.fabs(num_1/len(label_list))
    
def calculate_step_score(sample):
    step_score = []
    for i,step in enumerate(sample['instance'][:-1]):
        label_list = []
        for completion in step['completions']:
            label = check_answer(sample,completion)
            label_list.append(label)
            
        norm_score = normalize_score(label_list)
        norm_score = round(norm_score,3)
        step_score.append(norm_score)
        
    prompt = sample['instance'][-1]['prompt']
    label = float(check_answer(sample,prompt))
    norm_score = round(label,3)
    step_score.append(norm_score)
        
    return step_score

if __name__ == "__main__":
    
    with open("combined_data.json",'r') as f:
        data = json.load(f)[:]
        
    dataset_math = load_dataset("lighteval/MATH",name="all",split='train')
    dataset_math_test = load_dataset("lighteval/MATH",name="all",split='test')
    dataset_gsm = load_dataset("openai/gsm8k",name='main',split='train')
    
    dataset = []
    finish_idx = []
    for i in tqdm(range(len(data))):
        if "The answer is" in data[i]['prompt']:
            finish_idx.append(i)
            
    for i in tqdm(range(len(finish_idx))):
        if i == 0:
            instance = process_list(data[:finish_idx[i]+1])
        else:
            instance = process_list(data[finish_idx[i-1]+1:finish_idx[i]+1])
        dataset.append({"instance":instance})
    
    #dataset = dataset[:100]    
    filter_dataset = []
    for sample in tqdm(dataset):
        contain = False
        for ref in dataset_math:
            if ref['problem'] in sample['instance'][0]['prompt']:
                sample['task'] = "MATH"
                sample['ground_truth'] = ref['solution']
                contain = True
                filter_dataset.append(sample)
                break
        if contain == True:
            continue  
        for ref in dataset_gsm:
            if ref['question'] in sample['instance'][0]['prompt']:
                sample['task'] = "GSM"
                sample['ground_truth'] = ref['answer']
                filter_dataset.append(sample)
                contain = True
                break
        
            
        
    # count = 0
    # for sample in dataset:
    #     if "task" in sample:
    #         count += 1
            
    # print(count)
    dataset = filter_dataset
    print("-----------------")
    print("begin to label each process")
    print("-----------------")
    
    step_tag = 'ки' 
    store_data = []
    for sample in tqdm(dataset):
        step_score = calculate_step_score(sample)
        # answer_str = sample['instance'][-1]['prompt']
        
        # prompt_idx = answer_str.find("Step 1:")
        # prompt = answer_str[:prompt_idx]
        # answer = answer_str[prompt_idx:]
        # answer_list = answer.split(" ки\n")
            
        for i,ans in enumerate(sample['instance'][:-1]):
            store_data.append({"text":ans['prompt'],"label":float(step_score[i])})
        
        store_data.append({"text":sample['instance'][-1]['prompt'],"label":float(step_score[-1])})
        # for idx, answer in enumerate(step_score):
        #     answer_str = answer_str.replace(step_tag, str(answer), 1)
            
        # store_data.append(answer_str)
        
    with open("score_data.json",'w') as f:
        json.dump(store_data,f,indent=4,ensure_ascii=False)