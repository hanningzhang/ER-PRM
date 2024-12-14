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

def process_results(doc, completion, answer):
    if 'The answer is: ' in completion:
        split_ans = completion.split('The answer is: ')
        if len(split_ans) > 1:
            ans = split_ans[-1]
            # result = re.split(r'\.\n|\.\s', ans)
            # result = [x.strip() for x in result if x.strip()]
            # if result:
            #     extract_ans_temp = result[0]
            # else:
            #     extract_ans_temp = ans.split('.\n')[0]
                
            extract_ans_temp = check_and_remove_box(ans)
            extract_ans_temp = ans.split('.\n')[0]
            extract_ans_temp = ans.split('ки')[0]
            extract_ans_temp = extract_ans_temp.strip()
            if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
                extract_ans = extract_ans_temp[0:-1]
            else:
                extract_ans = extract_ans_temp
            extract_ans = extract_ans.strip()
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


def batch_data(data_list, batch_size=1):
    new_data_list = []
    for sample in data_list:
        # prompt = (
        #     f"Problem: {sample}\n"
        #     "Please provide a short and precise step-by-step solution for the question above in the following format, without any extra wording:\n"
        #     "Step 1: (logical step 1)\n"
        #     "Step 2: (logical step 2)\n"
        #     "...\n"
        #     "Step n: (logical last step)\n"
        #     "The answer is: (Final result)"
        #     "Please strictly stick to the format above."
        # )
        prompt = sample
        new_data_list.append(prompt)
    
    data_list = new_data_list
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def test_hendrycks_math(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('promt =====', problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            #temp_instr = problem_prompt.format(instruction=item["instruction"])
            temp_instr = item['problem']
            #temp_instr = item['instruction']
            hendrycks_math_ins.append(temp_instr)
            solution = item['solution']
            #solution = item['output']
            temp_ans = remove_boxed(util.last_boxed_only_string(solution))
            hendrycks_math_answers.append(temp_ans)

    #print(hendrycks_math_answers)
    print('total length ===', len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins[start:end]
    hendrycks_math_answers = hendrycks_math_answers[start:end]
    print('lenght ====', len(hendrycks_math_ins))
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048, stop=[])
    print('sampling =====', sampling_params)

    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size, enforce_eager=True, dtype = "float16", gpu_memory_utilization=0.9,swap_space=32,max_model_len=1024)
            
    res_completions = []
    for idx, (prompt, prompt_answer) in tqdm(enumerate(zip(batch_hendrycks_math_ins, hendrycks_math_answers))):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            prompt_temp = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    results = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        res = process_results(prompt, completion, prompt_answer)
        results.append(res)

    acc = sum(results) / len(results)
    #print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    #print('start===', start, ', end====',end)
    print('length====', len(results), ', acc====', acc)
    output_dir = args.output_dir.replace("/","_")
    with open(f"eval_result/{output_dir}",'w') as f:
        json.dump({"math":acc},f)
        json.dump(invalid_outputs, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='peiyi9979/mistral-7b-sft')  # model path
    parser.add_argument("--data_file", type=str, default='../data/test/MATH_test_500.jsonl')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=500)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("---------------")
    print("begin to evaluate the MATH dataset.")
    print("---------------")
    test_hendrycks_math(model=args.model, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size)
