import json
import re
from tqdm import tqdm
import stanza
from dataclasses import dataclass, field
from typing import Optional
from vllm import LLM, SamplingParams
from eval_gsm8k import is_number, extract_answer_number
from eval_math import remove_boxed, process_results
import util
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, pipeline


@dataclass
class ScriptArguments:
    completion_model_name_or_path: str = field(default="", metadata={"help": "the completion model name or path locally or from huggingface."})
    dataset_path: str = field(default="", metadata={"help": "dataset path for generator data."})
    output_dir: str = field(default="",metadata={"help":"location to store the PRM data."})
    tensor_parallel_size: int = field(default=2,metadata={"help":""})
    num_gpus: int = field(default=2)
    local_rank:int = field(default=0)
    reference_dataset: str = field(default="meta-math/MetaMathQA",metadata={"help":"Original dataset where the sample data comes from."})
    max_number: int = field(default=500,metadata={"help":"Max samples used to form reward model data."})

def check_math_answer(content,ground_truth):
    
    split_ans = content.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('ки')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        
        gt_ans = remove_boxed(util.last_boxed_only_string(ground_truth))
        #print(f"gt ans is {gt_ans}")
        # gt_ans = ground_truth.split('The answer is: ')
        # gt = gt_ans[-1]
        # extract_gt_temp = gt.split('ки')[0]
        # extract_gt_temp = extract_gt_temp.strip()
        # if len(extract_gt_temp)>0 and extract_gt_temp[-1] == '.':
        #     extract_gt = extract_gt_temp[0:-1]
        # else:
        #     extract_gt = extract_gt_temp
        # extract_gt = extract_gt.strip()
        if util.is_equiv(extract_ans, gt_ans):
            return True
        else:
            return False
    else:
        return False
    
def check_answer(sample,content,ground_truth):

    if "GSM" in sample['task']:
        temp_ans = sample['ground_truth'].split('#### ')[1]
        temp_ans = int(temp_ans.replace(',', ''))
        #print(float(extract_answer_number(sample['ground_truth'])))
        if not extract_answer_number(content):
            return 1
        if float(extract_answer_number(content)) == float(temp_ans):
            label = 0
        else:
            label = 1
    else:
        if check_math_answer(content,sample['ground_truth']):
            label = 0
        else:
            label = 1
            
    return label
    

def generate_completion(llm,sampling_params,sample,prompt,ground_truth):
    
    #label = None    ## 0 denotes good completion. 1 denotes bad completion.
    label_list = []
    if isinstance(prompt, list):
        pass
    else:
        prompt = [prompt]

    completions = llm.generate(prompt, sampling_params)
    completions_list = []
    for output in completions:
        prompt_temp = output.prompt
        generated_text = [output.outputs[i].text for i in range(len(output.outputs))]
        completions_list.append([prompt_temp,generated_text])

    for completion in completions_list:
        results = [check_answer(sample,content,ground_truth) for content in completion[1]]
    
        if 0 in results:
            label = 0
        else:
            label = 1
        
        label_list.append({"prompt":completion[0],"label":label})
        
    return label_list

if __name__ == "__main__":
    
    parser = HfArgumentParser((ScriptArguments))
    args = parser.parse_args_into_dataclasses()[0]
    
    downloaded = False
    while not downloaded:
        try:
            stanza.download('en')
            snlp = stanza.Pipeline(lang="en",processors='tokenize')
            downloaded = True
        except:
            print("not success in downloading stanza. Retrying....")
            time.sleep(2)

    stop_tokens = []
    sampling_params = SamplingParams(n=8, temperature=0.7, top_p=1, max_tokens=1024, stop=stop_tokens)
    print('sampling =====', sampling_params)
    llm = LLM(model=args.completion_model_name_or_path,tensor_parallel_size=args.tensor_parallel_size, enforce_eager=True, dtype = "float16",gpu_memory_utilization=0.9,swap_space=32)
    ## Load dataset
    with open(f"{args.dataset_path}/samples_{args.local_rank}.json",'r') as f:
        data = json.load(f)
    
    print("------------")
    print("begin to preprocess the sampling data")
    print("------------")
    
    dataset_math = load_dataset("lighteval/MATH",name="all",split='train')
    dataset_gsm = load_dataset("gsm8k",name='main',split='train')
    
    for sample in tqdm(data):
        for ref in dataset_math:
            if ref['problem'] in sample['prompt']:
                sample['task'] = "MATH"
                sample['ground_truth'] = ref['solution']
                break
        for ref in dataset_gsm:
            if ref['question'] in sample['prompt']:
                sample['task'] = "GSM"
                sample['ground_truth'] = ref['answer']
                break
    
    step_tag = 'ки' 
    print("------------")
    print("begin to label with markov process.")
    print("------------")
    save_data = []
    # new data to store the step with label  
    for sample in tqdm(data):
        prompt_list = []
        step_dict_list = []
        ground_truth = sample['ground_truth']
        question = sample['prompt']
        for answer in sample['answers']:          
            step_dict = {"question":question,
                     "answer":answer,
                     "ground_truth":ground_truth,
                     "step":[]}

            step_list = answer.split('\n')
            prompt = question + " "
        
            tmp = []
            for i, step in enumerate(step_list[:]):
                prompt += f"{step}\n"
                prompt_list.append(prompt)
                #label = generate_completion(llm,sampling_params,sample,prompt,ground_truth)
                tmp.append({"prompt":prompt,"step":step,"label":0})
            step_dict['step'] = tmp
            step_dict_list.append(step_dict)
            
        label_list = generate_completion(llm,sampling_params,sample,prompt_list,ground_truth)
        for current_step_dict in step_dict_list:
            for step_instance in current_step_dict["step"][:-1]:
                for label_instance in label_list:
                    if step_instance['prompt'] == label_instance['prompt']:
                        step_instance['label'] = label_instance['label']
                        break
                    
        for current_step_dict in step_dict_list:
            current_step_dict["step"][-1]['label'] = check_answer(sample,current_step_dict["step"][-1]['prompt'],ground_truth)
        
        ## We check the last step whether the answer is correct or not.
            # label = check_answer(sample,step_list[-1],ground_truth)
            # tmp.append({"step":step_list[-1],"label":label})
            
            # step_dict['step'] = tmp
        
        for current_step_list in step_dict_list:
            step_list = [i['step'] for i in current_step_dict['step']]
            
            save_dict = {"input": "","label":""}
            join_step = '\n'.join(step_list)
            save_dict['input'] = f"{question} {join_step}" 
            
            for i,step in enumerate(current_step_dict['step']):
                if step['label'] == 0:
                    step_list[i] = step_list[i].replace(step_tag,'+')
                else:
                    step_list[i] = step_list[i].replace(step_tag,'-')
                    
            join_step = '\n'.join(step_list)
            save_dict['label'] = f"{question} {join_step}" 
        
            save_data.append(save_dict)
        with open(f"{args.output_dir}/dataset_{args.local_rank}.json",'w') as f:
            json.dump(save_data,f,indent=4,ensure_ascii=False)
