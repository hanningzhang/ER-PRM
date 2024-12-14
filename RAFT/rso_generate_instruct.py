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

def format_gsm_dataset(raw_datasets):
    gsm_prompt = []
    math_prompt = []
    for sample in raw_datasets:
        if "GSM" in sample['type']:
            prompt = (
            f"Problem: {sample['query']}\n"
            "Please provide a short and precise step-by-step solution, and a numerical answer in the end, for the question above in the following format, without any extra wording:\n"
            "Step 1: (logical step 1)\n"
            "Step 2: (logical step 2)\n"
            "...\n"
            "Step n: (logical last step)\n"
            "The answer is: (Final result)."
            "Please strictly stick to the format above."
        )
            gsm_prompt.append(prompt)
    return gsm_prompt
    
def format_dataset(raw_datasets):
    formatted_dataset = []
    for sample in raw_datasets:
        prompt = (
            f"Problem: {sample['question']}\n"
            "Please provide a short and precise step-by-step solution, and a numerical answer in the end, for the question above in the following format, without any extra wording:\n"
            "Step 1: (logical step 1)\n"
            "Step 2: (logical step 2)\n"
            "...\n"
            "Step n: (logical last step)\n"
            "The answer is: (Final result)."
            "Please strictly stick to the format above."
        )
        formatted_dataset.append(prompt)
    return formatted_dataset

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
        for count,i in enumerate(doc_sents):
            text = f"Step {count+1}: {i}"
            truncate_doc.append(text)
        
        temp = " ки\n".join(truncate_doc)
        temp = temp + " ки"
        post_list.append(temp)
    return post_list

def save(store_data,args):
    with open(f"{args.output_dir}/samples_{args.local_rank}.json",'w') as f:
        json.dump(store_data,f,indent=4,ensure_ascii=False)

def extract_math_problem(prompt):
    start_marker = "Problem: "
    end_marker = "Please provide a short and precise step-by-step solution"
    
    # Find the start and end positions of the problem statement
    start_index = prompt.find(start_marker) + len(start_marker)
    end_index = prompt.find(end_marker)
    
    # Extract the problem statement
    if start_index != -1 and end_index != -1:
        problem_statement = prompt[start_index:end_index].strip()
        return problem_statement
    else:
        return ""

# def extract_content(text):
#     # Define the regex pattern to find content between [START] and [END]
#     pattern = re.compile(r'\[START\](.*?)\[END\]', re.DOTALL)
#     # Search for the pattern in the provided text
#     match = pattern.search(text)
#     if match:
#         # Return the matched content, stripping leading and trailing whitespace
#         text = match.group(1).strip()
#         text_list = text.split("\n")
#         response = ""
#         for step in text_list[:-2]:
#             response += f"{step} ки\n"
#         response += text_list[-2] + " "
#         idx = text_list[-1].find("Result: ")
#         if idx == 0:
#             final_answer = text_list[-1].replace("Result: ","The answer is ")
#             response += final_answer + " ки"
#         else:
#             response += text_list[-1] + " ки"
#         return response
#     else:
#         # Return an empty string if no match is found
#         return ""

def extract_content(text):
    #print(text)
    start_idx = text.find("Step 1:")
    if start_idx > 0:
        text = text[start_idx:]
    else:
        return ""
    
    text_list = text.split("\n")
    text_list = [i.strip() for i in text_list]
    new_list = []
    contain = False
    for i in text_list:
        if "The answer is" in i or "the answer is" in i:
            new_list.append(i)
            contain = True
            break
        else:
            new_list.append(i)
    
    if len(new_list) == 1:
        return f"{new_list[0]} ки"
    if contain:
        response = ""
        response = " ки\n".join(new_list[:-1])
        #response += new_list[-2] + " "
        match = re.search(r'\d+', new_list[-1])
        if match:
            ans = match.group()
            response += f"The answer is: {ans} ки"
        else:
            response += f"{new_list[-1]} ки"
    else:
        response = ""
        for step in new_list[:-1]:
            response += f"{step} ки\n"
        response += f"{new_list[-1]} ки"
        
    return response
           
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
    
    #raw_datasets = load_dataset(args.dataset)
    #raw_datasets = raw_datasets['train']
    raw_datasets = load_dataset("gsm8k",name='main',split='train')
    if args.sanity_check:
        prompt = format_gsm_dataset(raw_datasets)[args.current_iter*20:(args.current_iter+1)*20]
    else:
        #original version
        all_prompt = format_dataset(raw_datasets)
        #all_prompt = format_gsm_dataset(raw_datasets)
        #random.shuffle(all_prompt)
        prompt = all_prompt
        #prompt = all_prompt[args.current_iter*args.batch_size_per_iter:(args.current_iter+1)*args.batch_size_per_iter]
        prompt = prompt[int(len(prompt)*args.local_rank/args.num_gpus):int(len(prompt)*(args.local_rank+1)/args.num_gpus)]
    
    #print(prompt)  
    batch_prompt = batch_data(prompt, batch_size=args.batch_size)
    llm = LLM(model=args.model_name_or_path,tensor_parallel_size=args.tensor_parallel_size, dtype = "float16",enforce_eager=True, gpu_memory_utilization=0.85,swap_space=32,max_model_len=1024)
    tokenizer = llm.get_tokenizer()
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    sampling_params = SamplingParams(n=32, temperature=1, top_p=1, max_tokens=512, stop_token_ids=stop_token_ids, seed=args.random_seed)
    print('sampling =====', sampling_params)

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
            cleaned_rationales = []
            prompt = output.prompt
            question = prompt
            #question = extract_math_problem(prompt)
            if question == "":
                continue
            generated_text = [output.outputs[i].text for i in range(len(output.outputs))]
            
            for text in generated_text:
                extracted_rationale = extract_content(text)
                if extracted_rationale != "" and extracted_rationale.startswith('Step 1:'):
                    cleaned_rationales.append(extracted_rationale)
            # if args.current_iter == 0:
            #     answers = process_answer(args,generated_text)
            # else:
            #     answers = generated_text
            store_data.append({"prompt":question,"answers":cleaned_rationales})
            # if args.current_iter == 0:
            #     store_data.append({"prompt":question,"answers":cleaned_rationales})
            # else:
            #     store_data.append({"prompt":question,"answers":generated_text})
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
