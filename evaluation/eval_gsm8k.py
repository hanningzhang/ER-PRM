import argparse
import json
import re
import torch
from tqdm import tqdm
import jsonlines
from fraction import Fraction
from vllm import LLM, SamplingParams
import sys
import time
MAX_INT = sys.maxsize

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

def get_last_answer(completion):
    answer = ""
    temp = completion
    temp = temp.replace(",", "")
    temp = [s for s in re.findall(r'-?\d+\.?\d*', temp)]
    if len(temp) != 0:
        answer = temp[-1]
        if answer != "":
            if answer[-1] == ".":
                answer = answer[:-1]
            # round the answer to thenearest integer
            try:
                answer = str(round(float(answer)))
            except:
                answer = answer[:-1]
    if answer == "":
        return None
    return answer


def batch_data(data_list, batch_size=1):
    new_data_list = []
    for sample in data_list:
        # prompt = (
        #     f"Problem: {sample}\n"
        #     "Please provide a short and precise step-by-step solution, and a numerical answer in the end, for the question above in the following format, without any extra wording:\n"
        #     "Step 1: (logical step 1)\n"
        #     "Step 2: (logical step 2)\n"
        #     "...\n"
        #     "Step n: (logical last step)\n"
        #     "The answer is: (Final result)."
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


def gsm8k_test(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('promt =====', problem_prompt)
    with open(data_path,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            #temp_instr = problem_prompt.format(instruction=item["query"])
            temp_instr = item['query']
            gsm8k_ins.append(temp_instr)
            temp_ans = item['response'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('lenght ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)


    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size, enforce_eager=True, dtype = "float16", gpu_memory_utilization=0.9,swap_space=32,max_model_len=1024)
    tokenizer = llm.get_tokenizer()
    
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048,stop_token_ids=stop_token_ids)
    print('sampleing =====', sampling_params)
    
    result = []
    res_completions = []
    for idx, (prompt, prompt_answer) in tqdm(enumerate(zip(batch_gsm8k_ins, gsm8k_answers))):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    invalid_outputs = []
    valid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
        doc = {'question': prompt}
        y_pred = extract_answer_number(completion)
        if y_pred is None:
            y_pred = get_last_answer(completion)
        if y_pred != None:
            if float(y_pred) == float(prompt_answer):
                result.append(float(y_pred) == float(prompt_answer))
                temp = {'question': prompt, 'output': completion, 'pred':y_pred, 'answer': prompt_answer}
                valid_outputs.append(temp)
            else:
                result.append(False)
                temp = {'question': prompt, 'output': completion, 'pred':y_pred, 'answer': prompt_answer}
                invalid_outputs.append(temp)
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'pred':y_pred, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    #print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    #print('start===', start, ', end====', end)
    print('gsm8k length====', len(result), ', gsm8k acc====', acc)
    output_dir = args.output_dir.replace("/","_")
    with open(f"eval_result/{output_dir}",'w') as f:
        json.dump({"gsm8k":acc},f)
        json.dump(invalid_outputs, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default = 'peiyi9979/mistral-7b-sft')  # model path
    parser.add_argument("--data_file", type=str, default='../data/test/GSM8K_test.jsonl')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=1000)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    print("---------------")
    print("begin to evaluate the gsm8k dataset.")
    print("---------------")
    gsm8k_test(model=args.model, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size)
