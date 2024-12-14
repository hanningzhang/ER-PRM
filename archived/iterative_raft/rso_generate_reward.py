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

def format_dataset(gsm,math):
    prompt = []
    for sample in gsm:
        prompt.append(sample['question'])
    for sample in math:
        prompt.append(sample['problem'])
    return prompt

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

def process_answer(answer_list):
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
        post_list.append(" ".join(truncate_doc))
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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    downloaded = False
    while not downloaded:
        try:
            stanza.download('en')
            snlp = stanza.Pipeline(lang="en",processors='tokenize')
            downloaded = True
        except:
            print("not success in downloading stanza. Retrying....")
            time.sleep(2)
    
    dataset_math = load_dataset("lighteval/MATH",name="all",split='train')
    dataset_gsm = load_dataset("gsm8k",name='main',split='train')
    raw_datasets = format_dataset(dataset_gsm,dataset_math)
    if args.sanity_check:
        prompt = random.sample(raw_datasets,20)
    else:
        prompt = random.sample(raw_datasets,2000)
        prompt = prompt[int(len(prompt)*args.local_rank/args.num_gpus):int(len(prompt)*(args.local_rank+1)/args.num_gpus)]
    batch_prompt = batch_data(prompt, batch_size=args.batch_size)
    stop_tokens = []
    sampling_params = SamplingParams(n=8, temperature=0.85, top_p=1, max_tokens=1024, stop=stop_tokens)
    print('sampling =====', sampling_params)
    llm = LLM(model=args.model_name_or_path,tensor_parallel_size=args.tensor_parallel_size, enforce_eager=True, dtype = "float16", gpu_memory_utilization=0.9,swap_space=32)

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
            answers = process_answer(generated_text)
            store_data.append({"prompt":prompt,"answers":answers})
            #print(generated_text)
            
        count += 1
        if count % 1 == 0:
            save(store_data,args)
            
    save(store_data,args)
