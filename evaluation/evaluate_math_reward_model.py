from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import multiprocessing
from tqdm import tqdm
import argparse
import json
import time
import jsonlines
from MATH_CoT.evaluation.eval_gsm8k import extract_answer_number

import argparse
import json
import pdb
import torch
import jsonlines
from tqdm import tqdm
import util
from vllm import LLM, SamplingParams
import sys
import time
import re
MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

invalid_outputs = []

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
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None
    
def remove_text_box(s):
    left = "\\text{"
    idx = s.find(left)
    if idx >= 0:
        return s[len(left):-1]
    else:
        return s

def process_results(doc, completion, answer):
    if 'The answer is: ' in completion:
        split_ans = completion.split('The answer is: ')
        if len(split_ans) > 1:
            ans = split_ans[-1]
            extract_ans_temp = ans.split('ки')[0]
            extract_ans_temp = extract_ans_temp.strip()
            # result = re.split(r'\.\n|\.\s', ans)
            # result = [x.strip() for x in result if x.strip()]
            # if result:
            #     extract_ans_temp = result[0]
            # else:
            #     extract_ans_temp = ans.split('.\n')[0]
                
            # extract_ans_temp = check_and_remove_box(extract_ans_temp)
            # #extract_ans_temp = ans.split('.\n')[0]
            # #extract_ans_temp = ans.split('ки')[0]
            # extract_ans_temp = extract_ans_temp.strip()
            if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
                extract_ans = extract_ans_temp[0:-1]
            else:
                extract_ans = extract_ans_temp
            extract_ans = extract_ans.strip()
            extract_ans = remove_text_box(extract_ans)
            extract_ans = remove_left_right(extract_ans)
            if util.is_equiv(extract_ans, answer):
                return True
            else:
                temp = {'question': doc, 'output': completion, 'pred':extract_ans, 'answer': answer}
                invalid_outputs.append(temp)
                return False
        else:
            temp = {'question': doc, 'output': completion, 'pred':extract_ans, 'answer': answer}
            invalid_outputs.append(temp)
            return False
    else:
        split_ans = completion.split('Step')
        if len(split_ans) > 1:
            ans = split_ans[-1]
            extract_ans_temp = ans.split('ки')[0]
            extract_ans_temp = ans.split(':')[-1]
            extract_ans_temp = extract_ans_temp.strip()
            if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
                extract_ans = extract_ans_temp[0:-1]
            else:
                extract_ans = extract_ans_temp
            extract_ans = extract_ans.strip()
            extract_ans = remove_text_box(extract_ans)
            extract_ans = remove_left_right(extract_ans)
            if util.is_equiv(extract_ans, answer):
                return True
            else:
                temp = {'question': doc, 'output': completion, 'pred':extract_ans, 'answer': answer}
                invalid_outputs.append(temp)
                return False
        else:
            temp = {'question': doc, 'output': completion, 'pred':"", 'answer': answer}
            invalid_outputs.append(temp)
            return False
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_name_or_path", type=str, default='reward_llama_31_8b_autoregressive_ce5_1e-6lr_bs128')  # model path
    parser.add_argument("--dataset", type=str, default='verification')  # data path
    parser.add_argument("--output_dir", type=str, default="math_only")  # output dir
    parser.add_argument("--num_gpus", type=int, default=1)  # output dir
    parser.add_argument("--num_samples", type=int, default=1) 
    return parser.parse_args()

def batch_data(data_list, batch_size=8):
    n = batch_size
    batch_data = []
    for i in range(n-1):
        start = i * (len(data_list) // batch_size)
        end = (i+1)* (len(data_list) // batch_size)
        batch_data.append(data_list[start:end])

    last_start = (n-1) * (len(data_list) // batch_size)
    batch_data.append(data_list[last_start:len(data_list)])
    return batch_data

def select_sample(i,sample,model,tokenizer,candidate_tokens,step_tag_id,args):
    prompt = sample['prompt']
    scores_list = []
    text_list = []
    #sample_list = [prompt + sample['answers'][i] for i in range(len(sample['answers']))]
    sample_list = [prompt + sample['answers'][i] for i in range(args.num_samples)]
    sample_list = [i.replace(" ки\n"," ки") for i in sample_list]
    for j in range(0,len(sample_list),16):
        current_list = sample_list[j:j+16]
        input_id = torch.tensor(tokenizer(current_list,padding=True)['input_ids']).to(0)
        #print("test")
        with torch.no_grad():
            logits = model(input_id).logits[:,:,candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,0] 
            for i in range(scores.shape[0]):
                current = scores[i]
                step_scores = current[input_id[i] == step_tag_id]
                if len(step_scores):
                #scores_list.append(sum(step_scores)/len(step_scores))
                    scores_list.append(min(step_scores))
                    text_list.append([sample_list[i],min(step_scores).item()])
                else:
                    scores_list.append(0)
    idx = scores_list.index(max(scores_list))
    text = f"{prompt} {sample['answers'][idx]}"
        
    return text, text_list

def worker(gpu_id, model, tokenizer, data, output,finished_count,args):

    temp_instances = []
    good_token = '+'
    bad_token = '-'
    step_tag = 'ки'
    step_tag_id = tokenizer.encode(f" {step_tag}")[-1] 
    plus_tag_id = tokenizer.encode(' +')[-1]
    minus_tag_id = tokenizer.encode(' -')[-1]
    #candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]
    candidate_tokens = [plus_tag_id,minus_tag_id]
    for sample in tqdm(data):
        text, text_list = select_sample(gpu_id,sample,model,tokenizer,candidate_tokens,step_tag_id,args)
        temp_instances.append({"text":text,"text_list":text_list})
        
    # Save results
    output.put((gpu_id, temp_instances))
    finished_count.value += 1
       
if __name__ == "__main__":
    args = parse_args()

    torch.multiprocessing.set_start_method('spawn')
    tokenizer_list = []
    model_list = []
    print("---------------")
    print("begin to load reward model.")
    print("---------------")
    for i in range(args.num_gpus):
        downloaded = False
        while not downloaded:
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.reward_name_or_path)
                model = AutoModelForCausalLM.from_pretrained(args.reward_name_or_path, torch_dtype=torch.bfloat16).to(i).eval()
                downloaded = True
            except Exception as error:
                print("An error occurred:", error)
                print("Failed to load the reward model. Retrying....")
                time.sleep(2)

        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
        tokenizer_list.append(tokenizer)
        model_list.append(model)

    # with open(f"{args.dataset}/math_data.json",'r') as f:
    #     data = json.load(f)
    with open(f"reward_evaluate/data/shepherd_math_data_n64_seed42_maxlength.json",'r') as f:
        data = json.load(f)[:]

    LMFlow_data = {"type":"text_only","instances":[]}
    data_list = batch_data(data,batch_size = args.num_gpus)
    
    output = multiprocessing.Queue()
    
    finished_count = multiprocessing.Value('i', 0)
    processes = [
        multiprocessing.Process(target=worker, args=(i, model_list[i], tokenizer_list[i], data_list[i], output, finished_count,args))
        for i in range(args.num_gpus)
    ]
    print("---------------")
    print("begin to evaluate with the reward model.")
    print("---------------")
    for p in processes:
        p.start()

    while finished_count.value < len(processes):
        time.sleep(1)
        pass

    # Collect results
    results = [output.get() for _ in range(args.num_gpus)]
    
    sorted_results = sorted(results, key=lambda x: x[0])
    for res in sorted_results:
        LMFlow_data['instances'] += res[1]
    
    selected_data = [sample['text'] for sample in LMFlow_data['instances']]
    selected_candidates = [sample['text_list'] for sample in LMFlow_data['instances']]
    
    gt_data = []
    hendrycks_math_ins = []
    with open("data/test/MATH_test_500.jsonl","r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            #temp_instr = problem_prompt.format(instruction=item["instruction"])
            temp_instr = item['problem']
            #temp_instr = item['instruction']
            hendrycks_math_ins.append(temp_instr)
            solution = item['solution']
            #solution = item['output']
            temp_ans = remove_boxed(util.last_boxed_only_string(solution))
            temp_ans = remove_text_box(temp_ans)
            gt_data.append(temp_ans)
    
    selected_top5 = []
    for i,sample in enumerate(selected_candidates):
        sorted_list = sorted(sample, key=lambda x: x[1],reverse=True)
        selected_top5.append({"top_5":sorted_list[:5],"gt":gt_data[i]})
        
    total = len(data)
    count = 0   
    wrong_answers = []
    #print(gt_data)    
    for prompt,predict,gt in zip(hendrycks_math_ins,selected_data,gt_data):
        res = process_results(prompt, predict, gt)
        #print("test")
        if res:
            count += 1
        else:
            wrong_answers.append({"prompt":prompt,"prediction":predict,"gt":gt})
    
    print("--------------------")
    print(f"Reward Model Accuracy: {count/total}")
    print("--------------------")
    with open(f"{args.output_dir}/{args.reward_name_or_path}_sub_math_n{args.num_samples}.json",'w') as f:
        json.dump({"accuracy":count/total},f,indent=4,ensure_ascii=False)
        
    # with open(f"{args.output_dir}/{args.reward_name_or_path}_sub_math_n64_seed0_maxlength_wrong.json",'w') as f:
    #     json.dump(wrong_answers,f,indent=4,ensure_ascii=False)
        
    # with open(f"{args.output_dir}/{args.reward_name_or_path}_sub_math_n64_seed0_maxlength_wrong_top5.json",'w') as f:
    #     json.dump(selected_top5,f,indent=4,ensure_ascii=False)

