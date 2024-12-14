import argparse
import json
import re
import torch
import stanza
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset

def format_dataset_gsm(raw_datasets):
    prompt = []
    for sample in raw_datasets:
        prompt.append(sample['question'])
    return prompt

def format_dataset_math(raw_datasets):
    prompt = []
    for sample in raw_datasets:
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
    with open(args.output_dir,'w') as f:
        json.dump(store_data,f,indent=4,ensure_ascii=False)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default='peiyi9979/mistral-7b-sft')  # model path
    parser.add_argument("--dataset_gsm", type=str, default='gsm8k')  # data path
    parser.add_argument("--dataset_math", type=str, default='data/train/MATH_train.jsonl')  # data path
    parser.add_argument("--batch_size", type=int, default=256)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    parser.add_argument("--output_dir", type=str, default="rso_data/gsm_result.json")  # tensor_parallel_size
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    stanza.download('en')
    snlp = stanza.Pipeline(lang="en",processors='tokenize')
    
    raw_datasets = load_dataset(args.dataset_gsm,name='main')
    raw_datasets = raw_datasets['train']
    prompt = format_dataset_gsm(raw_datasets)
    #batch_prompt = batch_data(prompt, batch_size=args.batch_size)
    
    raw_datasets = []
    with open(args.dataset_math,'r') as f:
        json_list = list(f)
    for file in json_list:
        raw_datasets.append(json.loads(file))

    prompt += format_dataset_math(raw_datasets)
    print(len(prompt))
    batch_prompt = batch_data(prompt, batch_size=args.batch_size)
    
    stop_tokens = []
    sampling_params = SamplingParams(n=16, temperature=0.85, top_p=1, max_tokens=512, stop=stop_tokens)
    print('sampleing =====', sampling_params)
    llm = LLM(model=args.model_name_or_path,tensor_parallel_size=args.tensor_parallel_size, enforce_eager=True, gpu_memory_utilization=0.95)

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