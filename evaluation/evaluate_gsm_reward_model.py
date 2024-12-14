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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_name_or_path", type=str, default='reward_llama_31_8b_autoregressive_ce2_5e-7lr')  # model path
    parser.add_argument("--dataset", type=str, default='verification')  # data path
    parser.add_argument("--output_dir", type=str, default="gsm_only")  # output dir
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
            # logits = model(input_id).logits[:,:,candidate_tokens]
            # scores = logits.softmax(dim=-1)[:,:,0] 
            logits = model(input_id).logits
            logits = logits.softmax(dim=-1)
            logits = logits[:,:,candidate_tokens]
            scores = logits[:,:,0] 
            for i in range(scores.shape[0]):
                current = scores[i]
                step_scores = current[input_id[i] == step_tag_id]
                #print(step_scores)
                if len(step_scores):
                #scores_list.append(sum(step_scores)/len(step_scores))
                    scores_list.append(min(step_scores))
                    text_list.append([sample_list[i],min(step_scores).item()])
                else:
                    scores_list.append(0)
    idx = scores_list.index(max(scores_list))
    text = f"{prompt} {sample['answers'][idx]}"
    return text

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
    #step_tag_id = tokenizer.encode(f"{step_tag}")[-1] # 12902
    for sample in tqdm(data):
        text = select_sample(gpu_id,sample,model,tokenizer,candidate_tokens,step_tag_id,args)
        temp_instances.append({"text":text})
        
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

    # with open(f"{args.dataset}/gsm_data.json",'r') as f:
    #     data = json.load(f)
    with open(f"reward_evaluate/data/shepherd_gsm8k_data_n64_seed42_maxlength.json",'r') as f:
        data = json.load(f)

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
    
    gt_data = []
    with open("data/test/GSM8K_test.jsonl","r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_ans = item['response'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gt_data.append(temp_ans)
    
    total = len(data)
    count = 0       
    for predict,gt in zip(selected_data,gt_data):
        if not extract_answer_number(predict):
            continue
        if float(extract_answer_number(predict)):
            if float(extract_answer_number(predict)) == float(gt):
                count += 1
    
    print("--------------------")
    print(f"Reward Model Accuracy: {count/total}")
    print("--------------------")
    with open(f"{args.output_dir}/{args.reward_name_or_path}_ar_gsm_n{args.num_samples}.json",'w') as f:
        json.dump({"accuracy":count/total},f,indent=4,ensure_ascii=False)

