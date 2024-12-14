# This is the Python implementation for sampling rationales with LLaMA Policy Models

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
from transformers import AutoTokenizer

# specific format designed to extract step by step rationales from LLaMA 3/3.1 8B Instruct
def format_dataset(raw_datasets):
    formatted_dataset = []
    for sample in raw_datasets:
        prompt = (
            f"Problem: {sample['question']}\n"
            "Please provide a succinct step-by-step solution for the question above in the following format, without any extra wording, please put your final answer within \boxed{}:\n"
            "[START]\n"
            "Step 1: (logical step 1)\n"
            "Step 2: (logical step 2)\n"
            "...\n"
            "Step n: (logical last step)\n"
            "\boxed{answer}"
            "[END]\n"
            "Please strictly stick to the format above."
        )
        formatted_dataset.append(prompt)
    return formatted_dataset

def format_half_dataset(raw_datasets, tokenizer):
    gsm_prompt = []
    math_prompt = []
    for sample in raw_datasets:
        if "GSM" in sample['type']:
            chat = [
                {"role": "system", "content": (
                    "For the given math question from user, please provide a succinct step-by-step solution for the question in the following format, without any extra wording:\n"
                    "[START]\n"
                    "Step 1: (logical step 1)\n"
                    "Step 2: (logical step 2)\n"
                    "...\n"
                    "Step n: (logical last step)\n"
                    "The answer is: (Final Answer)\n"
                    "[END]\n"
                    "Please strictly stick to the format above."
                    )
                },
                {"role": "user", "content": sample['query']},
            ]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False)
            gsm_prompt.append(prompt)
        else:
            chat = [
                {"role": "system", "content": (
                    "For the given math question from user, please provide a succinct step-by-step solution for the question in the following format and put your final answer within \\boxed{}, without any extra wording:\n"
                    "[START]\n"
                    "Step 1: (logical step 1)\n"
                    "Step 2: (logical step 2)\n"
                    "...\n"
                    "Step n: (logical last step)\n"
                    "The answer is: \\boxed{final answer}\n"
                    "[END]\n"
                    "Please strictly stick to the format above."
                    )
                },
                {"role": "user", "content": sample['query']},
            ]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False)
            math_prompt.append(prompt)
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
    with open(f"{args.ds_type}_dpo_prep_rationale.json",'w+', encoding='utf-8') as f:
        json.dump(store_data,f,indent=4,ensure_ascii=False)


def extract_content(text):
    start_marker = "<|start_header_id|>assistant<|end_header_id|>"
    start_idx = text.find(start_marker) + len(start_marker)
    if start_idx != -1:
        return text[start_idx:].replace("[START]","").replace("[END]","").strip()
    else:
        return ""


def extract_math_problem(prompt):
    start_marker = "<|start_header_id|>user<|end_header_id|>"
    end_marker = "<|eot_id|>"
    
    # Find the start and end positions of the problem statement
    start_index = prompt.find(start_marker) + len(start_marker)
    
    # Extract the problem statement
    if start_index != -1:
        problem_statement = prompt[start_index:-len(end_marker)].strip()
        return problem_statement
    else:
        return ""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default='meta-llama/Llama-3.1-8B-Instruct')  # model path
    parser.add_argument("--dataset", type=str, default='meta-math/MetaMathQA')  # data path, default to use GSM8K
    parser.add_argument("--batch_size", type=int, default=1024)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=2)  # tensor_parallel_size
    #parser.add_argument("--output_dir", type=str, default="rso_data")  # output location
    #parser.add_argument("--current_iter", type=int, default=0)  # current iteration
    parser.add_argument("--num_gpus",type=int,default=2)
    #parser.add_argument("--local_rank",type=int,default=0)
    #parser.add_argument("--batch_size_per_iter", type=int, default=4096)  # number of samples per iteration
    #parser.add_argument("--sanity_check", type=int, default=0)  # sanity check
    parser.add_argument("--random_seed", type=int, default=42)  # random seed
    parser.add_argument("--sample_n", type=int, default=16) # number of sampling times for each prompt
    parser.add_argument("--ds_type", type=str, default='gsm8k')  # data path, default to use GSM8K
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    random.seed(args.random_seed)

    # load a tokenizer to use the chat template for math
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    raw_datasets = load_dataset(args.dataset, "main")
    raw_datasets = raw_datasets['train']
    if args.sanity_check:
        #prompt = format_dataset(raw_datasets)[args.current_iter*20:(args.current_iter+1)*20]
        exit()
    else:
        #half GSM8K and half MATH
        gsm_all_prompt, math_all_prompt = format_half_dataset(raw_datasets, tokenizer)
        
        gsm_selected_prompt = random.sample(gsm_all_prompt, 10000)
        math_selected_prompt = random.sample(math_all_prompt, 10000)

        if args.ds_type == 'gsm8k':
            prompt = gsm_selected_prompt
        elif args.ds_type == 'math':
            prompt = math_selected_prompt
        else:
            print("No such DS")
            exit()

    batch_prompt = batch_data(prompt, batch_size=args.batch_size)
    stop_tokens = []
    sampling_params = SamplingParams(n=args.sample_n, temperature=0.9, top_p=0.9, max_tokens=1024, stop=stop_tokens, seed=args.random_seed)
    print('sampling =====', sampling_params)
    llm = LLM(model=args.model_name_or_path,tensor_parallel_size=args.tensor_parallel_size, dtype = "float16",enforce_eager=True, gpu_memory_utilization=0.95,swap_space=32)

    print("---------------")
    print("begin to sampling from the SFT model")
    print("---------------")
    
    store_data = []
    count = 0
    for idx, prompt in tqdm(enumerate(batch_prompt), total=len(batch_prompt)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            cleaned_rationales = []
            prompt = output.prompt
            generated_text = [output.outputs[i].text for i in range(len(output.outputs))]
            
            # extract the math problem
            question = extract_math_problem(prompt)
            if question == "":
                continue

            # process and extract the rationale
            for text in generated_text:
                extracted_rationale = extract_content(text)
                if extracted_rationale != "" and extracted_rationale.startswith('Step 1:'):
                    cleaned_rationales.append(extracted_rationale)
            
            store_data.append({"prompt":question, "answers":cleaned_rationales})
        count += 1
        if count % 1 == 0:
            save(store_data, args)
    
    print("---------------")
    print("Successfully sampled from the SFT model")
    print("Now begin to save the data")
    print("---------------")       
    save(store_data,args)
    print("---------------")
    print("Saved the sampling data successfully!")
    print("---------------")
