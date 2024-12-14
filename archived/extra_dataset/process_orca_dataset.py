from IPython.display import display
from datasets import load_dataset
import stanza
import numpy
import torch
from tqdm import tqdm
import concurrent
import concurrent.futures
import time
import json
import argparse


def check_step(list):
    new_list = []
    for sample in list:
        if ". ки" in sample:
            new_list.append(sample)
        else:
            new_text = sample.replace(" ки",". ки")
            new_list.append(new_text)
    return new_list


def filter_out(ds):
    problem_step_by_step = set()
    problem_step = set()
    problem_step1 = set()

    for idx, text in enumerate(ds['train']['answer']):
        if 'step by step' in text.lower():
            problem_step_by_step.add(idx)

    for idx, text in enumerate(ds['train']['answer']):
        if 'step' in text.lower():
            problem_step.add(idx)

    for idx, text in enumerate(ds['train']['answer']):
        if 'step 1' in text.lower():
            problem_step1.add(idx)

    final_result = problem_step_by_step.intersection(problem_step).intersection(problem_step1)
    return list(final_result)


def process_orca_dataset(question_list,answer_list,snlp, excluded_question):
    step_tag = 'ки' 
    dataset = []

    for i in range(len(question_list)):
        if i in excluded_question:
            continue
        answer = answer_list[i]
        answer = answer.replace("\n\n", ". ")
        answer = answer.replace("\n", ". ")
        doc = snlp(answer)
        sentence_list = [sentence.text for sentence in doc.sentences]
        sentence_list = [i.replace("\n"," ") for i in sentence_list]
        if len(sentence_list) < 2:
            continue
        sentence_length = len(sentence_list)
        step_list = []
        
        for j in range(sentence_length):
            step_list.append(f"Step {j+1}: {sentence_list[j]} {step_tag}\n")
        
        step_list = check_step(step_list)  
        first_steps = ''.join(step_list)

        dataset.append({"prompt":question_list[i],"answer":first_steps})

    return dataset


# multi thread
def process_dataset_parallel(ds, question_list, answer_list, num_workers):
    st = time.time()
    final_datasets = []
    excluded_question = filter_out(ds)

    try:
        stanza.download('en')
        snlp = stanza.Pipeline(lang="en",processors='tokenize')
    except:
        print("stanza failed")
        return None

    if len(question_list) < num_workers:
        work_size = 1
    else:
        work_size = int(len(question_list) / num_workers)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for idx in range(0, len(question_list), work_size):
            if len(question_list) - idx > work_size:
                futures.append(executor.submit(process_orca_dataset, question_list[idx:idx + work_size], answer_list[idx:idx + work_size], snlp, excluded_question))
            else:
                futures.append(executor.submit(process_orca_dataset, question_list[idx:len(question_list)], answer_list[idx:len(answer_list)], snlp, excluded_question))
        
        for future in concurrent.futures.as_completed(futures):
            for item in future.result():
                final_datasets.append(item)
    
    for idx in excluded_question:
        final_datasets.append({"prompt":question_list[idx],"answer":answer_list[idx]})

    end = time.time()
    print("Execution time: ", end-st, " seconds")

    return final_datasets


def main():
    parser = argparse.ArgumentParser(description="Uncertainty_Generation")
    parser.add_argument("--num_workers", type=int, default=8, choices=[1,4,8], help="number of worker thread")
    parser.add_argument(
        "--save_path", type=str, default="orca_processed.json", help="path to save the processed orca dataset"
    )
    args = parser.parse_args()
    
    ds = load_dataset("microsoft/orca-math-word-problems-200k")
    final_processed_orca_ds = process_dataset_parallel(ds, ds['train']['question'], ds['train']['answer'], args.num_workers)

    with open(args.save_path, 'w') as f:
        json.dump(final_processed_orca_ds, f, indent=4)


if __name__ == "__main__":
    main()
