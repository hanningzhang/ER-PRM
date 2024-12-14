# This is the Python implementation for sampling step by step rationales LLaMA Policy Models

import argparse
import json
import re
import torch
import stanza
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset
import time
import random

def format_dataset(raw_datasets):
    prompt = []
    for sample in raw_datasets:
        prompt.append(sample['query'])
    return prompt

def format_half_dataset(raw_datasets):
    gsm_prompt = []
    math_prompt = []
    for sample in raw_datasets:
        if "GSM" in sample['type']:
            gsm_prompt.append(sample['query'])
        else:
            math_prompt.append(sample['query'])
    return gsm_prompt, math_prompt

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    batch_data.append(data_list[last_start:len(data_list)])
    return batch_data

def process_answer(args,answer_list):
    post_list = []
    for sample in answer_list:
        doc = snlp(sample)
        doc_sents = [sentence.text for sentence in doc.sentences]
        
        truncate_doc = []
        for i in doc_sents:
            if "The answer is" in i:
                truncate_doc.append(i)
                break
            else:
                truncate_doc.append(i)
        
        if "Llama-3" in args.model_name_or_path:
            post_list.append(" ".join(truncate_doc))
        else:
            temp = " ".join(truncate_doc)
            temp = temp.replace(" ки "," ки\n")
            post_list.append(temp)
    return post_list

def save(store_data,args):
    with open(f"{args.output_dir}/samples_{args.local_rank}.json",'w') as f:
        json.dump(store_data,f,indent=4,ensure_ascii=False)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default='peiyi9979/mistral-7b-sft')  # model path
    parser.add_argument("--dataset", type=str, default='meta-math/MetaMathQA')  # data path
    parser.add_argument("--batch_size", type=int, default=1024)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=2)  # tensor_parallel_size
    parser.add_argument("--output_dir", type=str, default="rso_data")  # output location
    parser.add_argument("--current_iter", type=int, default=0)  # current iteration
    parser.add_argument("--num_gpus",type=int,default=2)
    parser.add_argument("--local_rank",type=int,default=0)
    parser.add_argument("--batch_size_per_iter", type=int, default=4096)  # number of samples per iteration
    parser.add_argument("--sanity_check", type=int, default=0)  # sanity check
    parser.add_argument("--random_seed", type=int, default=42)  # random seed
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    downloaded = False
    while not downloaded:
        try:
            stanza.download('en')
            snlp = stanza.Pipeline(lang="en",processors='tokenize')
            downloaded = True
        except Exception as error:
            print("An error occurred:", error)
            time.sleep(2)
    
    raw_datasets = load_dataset(args.dataset)
    raw_datasets = raw_datasets['train']
    if args.sanity_check:
        prompt = format_dataset(raw_datasets)[args.current_iter*20:(args.current_iter+1)*20]
    else:
        #original version
        all_prompt = format_dataset(raw_datasets)
        #random.shuffle(all_prompt)
        prompt = all_prompt[args.current_iter*args.batch_size_per_iter:(args.current_iter+1)*args.batch_size_per_iter]
        prompt = prompt[int(len(prompt)*args.local_rank/args.num_gpus):int(len(prompt)*(args.local_rank+1)/args.num_gpus)]
        
        #half GSM8K and half MATH
        # gsm_all_prompt, math_all_prompt = format_half_dataset(raw_datasets)
        # gsm_prompt = gsm_all_prompt[int(args.current_iter*args.batch_size_per_iter/2):int((args.current_iter+1)*args.batch_size_per_iter/2)]
        # gsm_prompt = gsm_prompt[int(len(gsm_prompt)*args.local_rank/args.num_gpus):int(len(gsm_prompt)*(args.local_rank+1)/args.num_gpus)]
        
        # math_prompt = math_all_prompt[int(args.current_iter*args.batch_size_per_iter/2):int((args.current_iter+1)*args.batch_size_per_iter/2)]
        # math_prompt = math_prompt[int(len(math_prompt)*args.local_rank/args.num_gpus):int(len(math_prompt)*(args.local_rank+1)/args.num_gpus)]
        
        # prompt = gsm_prompt + math_prompt
    
    # prompt = []
    # dataset_math = load_dataset("lighteval/MATH",name="all",split='train')
    # dataset_gsm = load_dataset("gsm8k",name='main',split='train')
    # for sample in dataset_gsm:
    #     prompt.append(sample['question'])
    # # for sample in dataset_math:
    # #     prompt.append(sample['problem'])
    # if args.current_iter %2 == 0:
    #     prompt = prompt[:int(len(prompt)/2)]
    # else:
    #     prompt = prompt[int(len(prompt)/2):]
    # prompt = prompt[int(len(prompt)*args.local_rank/args.num_gpus):int(len(prompt)*(args.local_rank+1)/args.num_gpus)]
    
    batch_prompt = batch_data(prompt, batch_size=args.batch_size)
    stop_tokens = []
    sampling_params = SamplingParams(n=16, temperature=1, top_p=1, max_tokens=512, stop=stop_tokens, seed=args.random_seed)
    print('sampling =====', sampling_params)
    llm = LLM(model=args.model_name_or_path,tensor_parallel_size=args.tensor_parallel_size, dtype = "float16",enforce_eager=True, gpu_memory_utilization=0.95,swap_space=32)

    print("---------------")
    print("begin to sampling from the SFT model")
    print("---------------")
    
    store_data = []
    count = 0
    for idx, prompt in tqdm(enumerate(batch_prompt),total=len(batch_prompt)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            #print(len(output.outputs))
            prompt = output.prompt
            generated_text = [output.outputs[i].text for i in range(len(output.outputs))]
            #answers = process_answer(args,generated_text)
            answers = generated_text
            store_data.append({"prompt":prompt,"answers":answers})
            #print(generated_text)
            
        count += 1
        if count % 1 == 0:
            save(store_data,args)
    
    print("---------------")
    print("Successfully sampled from the SFT model")
    print("Now begin to save the data")
    print("---------------")       
    save(store_data,args)
    print("---------------")
    print("Saved the sampling data successfully!")
    print("---------------")
