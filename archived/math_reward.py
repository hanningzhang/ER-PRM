import gc
import argparse
import json
import re
import random
import stanza
import accelerate
from tqdm import tqdm
from datasets import load_dataset, Dataset
from dataclasses import dataclass, field
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    HfArgumentParser,
    Trainer,
    default_data_collator
)


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(default='mistralai/Mistral-7B-v0.1')  # sft model
    use_flash_attention: bool = field(default=True) 
    dataset_path: str = field(default="peiyi9979/Math-Shepherd")
    block_size: int =field(default=1024)
    current_iteration: int = field(default=1)
    total_iteration: int = field(default=3)

if __name__ == "__main__":

    parser = HfArgumentParser((TrainingArguments,ScriptArguments))
    training_args,script_args = parser.parse_args_into_dataclasses()

    #training_args.learning_rate *= (script_args.total_iteration-script_args.current_iteration+1)/script_args.total_iteration
     
    downloaded = False
    print("---------------")
    print("begin to load the base model")
    print("---------------")
    while not downloaded:
        try:
            model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, 
                                                 torch_dtype=torch.bfloat16,
                                                 use_flash_attention_2=True if script_args.use_flash_attention else None)
            tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
            downloaded = True
        except Exception as error:
            print("An error occurred:", error)
            print("Failed to load the SFT model. Retrying....")
            time.sleep(2)

    # with open(f"{script_args.dataset_path}/reward_dataset.json",'r') as f:
    #     raw_datasets = json.load(f)
    raw_datasets = load_dataset(script_args.dataset_path,split='train')
    # raw_datasets = raw_datasets['train'][:]
    
    raw_dict = {
        "input":[],
        "label":[]
    }
    dataset_math = load_dataset("lighteval/MATH",name="all",split='train')
    train_set = []
    all_question = []
    for i,sample in tqdm(enumerate(raw_datasets)):
        if "MATH" in sample['task']:
            idx = raw_datasets[i]['input'].find(" Step 1:")
            prompt = raw_datasets[i]['input'][:idx]
            if prompt in all_question:
                continue
            else:
                all_question.append(prompt)
    print(f"The length of the all question: {len(all_question)}")   
    #random.shuffle(all_question)        
    for all_q in tqdm(all_question[:]):
        for ref in dataset_math:
            if ref['problem'] in all_q:
                train_set.append(all_q)
                break
            
    print(f"The length of train set: {len(train_set)}")
            
    for i,sample in tqdm(enumerate(raw_datasets)):
        if "GSM" in sample['task']:
            raw_dict['input'].append(raw_datasets[i]['input'])
            raw_dict['label'].append(raw_datasets[i]['label'])
        else:
            idx = raw_datasets[i]['input'].find(" Step 1:")
            prompt = raw_datasets[i]['input'][:idx]
            if prompt in train_set:
                raw_dict['input'].append(raw_datasets[i]['input'])
                raw_dict['label'].append(raw_datasets[i]['label'])
                    
    
    print(len(raw_dict['input']))
    raw_datasets = raw_dict   
    with open("filtered_shepherd_dataset.json",'w') as f:
        json.dump(raw_datasets,f,indent=4)
    step_tag = 'ки'
    step_tag_id = tokenizer.encode(f"{step_tag}")[-1] # 12902
    plus_tag_id = tokenizer.encode('+')[-1]
    minus_tag_id = tokenizer.encode('-')[-1]
    
    dataset_dict = {
        "input_ids":[],
        "attention_mask":[],
        "labels":[]
    }
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    #tokenizer.padding_side = "left"
        
    print(tokenizer.pad_token)
    
    print("---------------")
    print("begin to process the dataset")
    print("---------------")
    
    # raw version
    # for i in tqdm(range(len(raw_datasets['input']))):
    #     raw_text = raw_datasets['input'][i]
    #     encode = tokenizer(raw_text,
    #                        add_special_tokens=True,
    #                         truncation=True)
    #     dataset_dict['input_ids'].append(encode['input_ids'])
    #     dataset_dict['attention_mask'].append(encode['attention_mask'])
    #     labels = encode['input_ids'].copy()
    #     reference_labels = tokenizer(raw_datasets['label'][i])['input_ids']
    #     for i in range(len(labels)):
    #         if labels[i] == step_tag_id:
    #             labels[i] = reference_labels[i]
    #         else:
    #             labels[i] = -100
    #     dataset_dict['labels'].append(labels)
        
    # shift version
    if "Llama-3" in script_args.model_name_or_path:
        step_tag = 'ки'
        step_tag_id = tokenizer.encode(f" {step_tag}")[-1] 
        plus_tag_id = tokenizer.encode(' +')[-1]
        minus_tag_id = tokenizer.encode(' -')[-1]
        for i in tqdm(range(len(raw_datasets['input']))):
            raw_text = raw_datasets['input'][i]
            raw_text = raw_text.replace(" ки\n"," ки ")
            encode = tokenizer(raw_text,
                           add_special_tokens=True,
                            truncation=True)
            dataset_dict['input_ids'].append(encode['input_ids'])
            dataset_dict['input_ids'][i].extend([tokenizer.pad_token_id])
        
            dataset_dict['attention_mask'].append(encode['attention_mask'])
            dataset_dict['attention_mask'][i].extend([0])
            labels = encode['input_ids'].copy()
            raw_label = raw_datasets['label'][i]
            raw_label = raw_label.replace(" +\n"," + ")
            raw_label = raw_label.replace(" -\n"," - ")
            reference_labels = tokenizer(raw_label)['input_ids']
            reference_labels = [tokenizer.pad_token_id] + reference_labels
            assert len(reference_labels) == len(dataset_dict['input_ids'][i])
            for i in range(len(reference_labels)):
                if reference_labels[i] == plus_tag_id or reference_labels[i] == minus_tag_id:
                    continue
                else:
                    reference_labels[i] = -100
            dataset_dict['labels'].append(reference_labels)
    else:
        for i in tqdm(range(len(raw_datasets['input']))):
            raw_text = raw_datasets['input'][i]
            encode = tokenizer(raw_text,
                           add_special_tokens=True,
                            truncation=True)
            dataset_dict['input_ids'].append(encode['input_ids'])
            dataset_dict['input_ids'][i].extend([tokenizer.pad_token_id])
        
            dataset_dict['attention_mask'].append(encode['attention_mask'])
            dataset_dict['attention_mask'][i].extend([0])
            labels = encode['input_ids'].copy()
            reference_labels = tokenizer(raw_datasets['label'][i])['input_ids']
            reference_labels = [tokenizer.pad_token_id] + reference_labels
            assert len(reference_labels) == len(encode['input_ids'])
            for i in range(len(reference_labels)):
                if reference_labels[i] == plus_tag_id or reference_labels[i] == minus_tag_id:
                    continue
                else:
                    reference_labels[i] = -100
            dataset_dict['labels'].append(reference_labels)
    
    for i in tqdm(range(len(dataset_dict['input_ids']))):
        block_size = script_args.block_size
        max_length = min(block_size, tokenizer.model_max_length)
        pad_length = max_length - len(dataset_dict["input_ids"][i])
        if pad_length < 0:
            # Truncates too long samples
            for key in ["input_ids", "attention_mask", "labels"]:
                dataset_dict[key][i] = dataset_dict[key][i][:pad_length]
        else:
            # Pads too short samples
            pad_token_id = tokenizer.pad_token_id
            dataset_dict["input_ids"][i].extend(
                [pad_token_id for _ in range(pad_length)]
            )
            dataset_dict["attention_mask"][i].extend(
                [0 for _ in range(pad_length)]
            )
            dataset_dict["labels"][i].extend(
                [-100 for _ in range(pad_length)]
            )
    
    #print(dataset_dict['input_ids'][0])                   
    dataset = Dataset.from_dict(dataset_dict)
    
    data_collator = default_data_collator
    finetuner = Trainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=default_data_collator,
    )
    print("---------------")
    print("begin to fine-tune the reward model")
    print("---------------")
    finetuner.train()

    print("---------------")
    print("Finish training the reward model!")
    print("Now begin to save the model.")
    print("---------------")
    finetuner.save_model(training_args.output_dir)
    finetuner.tokenizer.save_pretrained(training_args.output_dir)
    print("---------------")
    print("Saved the reward model successfully.")
    print("---------------")
