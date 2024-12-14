from datasets import load_dataset
import stanza
from tqdm import tqdm
import concurrent
import concurrent.futures
import time
import json
import argparse
import random
import re


def check_step(list):
    new_list = []
    for sample in list:
        if ". ки" in sample:
            new_list.append(sample)
        else:
            new_text = sample.replace(" ки",". ки")
            new_list.append(new_text)
    return new_list


def is_chinese(string):
    """
    Check if the string contain Chinese
    :param string
    :return: bool
    """
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True


def numina_question_filter(numina_ds):
    final_ds = []
    filter_out_list = {'chinese': [], 'http': [], 'multiple_choice': [], 'statement_choice': [], 'table_qes': [], 'subquestion': [],
                    'code': []}
    for idx, question in enumerate(numina_ds['train']):
        if question['source'] in ('gsm8k', 'math'):
            continue
        if is_chinese(question['problem']) or is_chinese(question['solution']):
            filter_out_list['chinese'].append(idx)
            continue
        elif "http" in question['problem'] or "http" in question['solution']:
            filter_out_list['http'].append(idx)
            continue
        elif "A" in question['problem'] and "B" in question['problem'] and "C" in question['problem'] and "D" in question['problem']:
            filter_out_list['multiple_choice'].append(idx)
            continue
        elif "Among the following statements" in question['problem']:
            filter_out_list['statement_choice'].append(idx)
            continue
        elif "------" in question['problem']:
            filter_out_list['table_qes'].append(idx)
            continue
        elif "(1)" in question['problem'] and "(2)" in question['problem']:
            filter_out_list['subquestion'].append(idx)
            continue
        elif "(i)" in question['problem'] and "(ii)" in question['problem'] and "following conditions" not in question['problem'] and "following properties" not in question['problem']:
            filter_out_list['subquestion'].append(idx)
            continue
        elif "[asy]" in question['problem']:
            filter_out_list['code'].append(idx)
            continue
        final_ds.append(question)
    for key, val in filter_out_list.items():
        print(key, len(val))
    return final_ds


def process_numina_ds(numina_ds, snlp):
    # if size is larger than metamathQA, then need to reduce size, take half from each class of questions
    if len(numina_ds) > 400000:
        temp_numina_ds = []
        question_class_dict = {}
        for idx, question in tqdm(enumerate(numina_ds)):
            if question['source'] in question_class_dict:
                question_class_dict[question['source']].append(idx)
            else:
                question_class_dict[question['source']] = []
        
        for source, index_list in question_class_dict.items():
            for i in tqdm(index_list[:len(index_list) // 2]):
                temp_numina_ds.append(numina_ds[i])
    random.shuffle(temp_numina_ds)
    print(len(temp_numina_ds))

    print("START PROCESS NUMINA")
    dataset = []
    step_tag = 'ки' 

    for question in tqdm(temp_numina_ds):
        answer = question['solution']

        lower_cased_answer = answer.lower()
        # the answer alreday contains numerical steps
        if "1." in lower_cased_answer and "2. " in lower_cased_answer:
            sentence_list = split_string_by_numerical_list(answer)
            if len(sentence_list) < 2:
                continue
            step_list = []
            for j in range(len(sentence_list)):
                if contains_numbered_list_item(sentence_list[j]):
                    step_list.append(f"Step {sentence_list[j]} {step_tag}\n")
                else:
                    step_list.append(f"{sentence_list[j]} {step_tag}\n")
            
            step_list = check_step(step_list)
            first_steps = ''.join(step_list)
            dataset.append({"source":question['source'],"prompt":question['problem'],"answer":first_steps})
        # no numerical list, use stanza to split sentence
        else:
            answer = answer.replace("\n\n", ". ")
            answer = answer.replace("\n", ". ")
            doc = snlp(answer)
            sentence_list = [sentence.text for sentence in doc.sentences]
            sentence_list = [i.replace("\n"," ") for i in sentence_list]
            if len(sentence_list) < 2:
                continue
            step_list = []
            
            for j in range(len(sentence_list)):
                step_list.append(f"Step {j+1}: {sentence_list[j]} {step_tag}\n")
            
            step_list = check_step(step_list)  
            first_steps = ''.join(step_list)

            dataset.append({"source":question['source'],"prompt":question['problem'],"answer":first_steps})
    return dataset


# split the sentences that contain 1. 2. 3 numerical list
def split_string_by_numerical_list(input_string):
    # Define the regular expression pattern to match the numerical list items at the beginning of a line
    pattern = re.compile(r'(?<=\n)\d+\.\s(?=[A-Z])')

    # Find all positions where the numerical list items start
    positions = [match.start() for match in pattern.finditer(input_string)]
    
    # Add the start and end of the string to the positions list
    positions.insert(0, 0)
    positions.append(len(input_string))

    # Split the string based on the positions
    sentences = [input_string[positions[i]:positions[i + 1]].strip() for i in range(len(positions) - 1)]

    return sentences


# detect if a sentence is start with a numerical order
def contains_numbered_list_item(sentence):
    # Define the regex pattern to match a number followed by a dot and a space at the start of the sentence
    pattern = r'^\s*\d+\.\s'
    # Use re.search to find the pattern in the sentence
    match = re.search(pattern, sentence)
    # Return True if a match is found, otherwise False
    return bool(match)


def main():
    parser = argparse.ArgumentParser(description="Uncertainty_Generation")
    parser.add_argument("--num_workers", type=int, default=8, choices=[1,4,8], help="number of worker thread")
    parser.add_argument(
        "--save_path", type=str, default="numina_processed.json", help="path to save the processed orca dataset"
    )
    parser.add_argument("--random_seed", type=int, default=42, help="set random seed")
    args = parser.parse_args()

    random.seed(args.random_seed)

    try:
        stanza.download('en')
        snlp = stanza.Pipeline(lang="en",processors='tokenize')
    except:
        print("stanza failed")
        return None
    
    numina_ds = load_dataset("AI-MO/NuminaMath-CoT")
    filtered_numina_ds = numina_question_filter(numina_ds)
    final_numina_ds = process_numina_ds(filtered_numina_ds, snlp)
    # final_processed_orca_ds = process_dataset_parallel(ds, ds['train']['question'], ds['train']['answer'], args.num_workers)

    with open(args.save_path, 'w') as f:
        json.dump(final_numina_ds, f, indent=4)


if __name__ == "__main__":
    main()