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
import numpy as np

import torch
#from vllm import LLM, SamplingParams
#from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    HfArgumentParser,
    Trainer
)

@dataclass
class ScriptArguments:
    model_name_or_path: str = field(default='peiyi9979/mistral-7b-sft')  # sft model
    use_flash_attention: bool = field(default=True) 
    dataset_path: str = field(default="LMFlow/metamath_sft.json")
    current_iteration: int=field(default=1)
    total_iteration: int=field(default=10)
    random_seed: int=field(default=42)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     
if __name__ == "__main__":

    parser = HfArgumentParser((TrainingArguments,ScriptArguments))
    training_args,script_args = parser.parse_args_into_dataclasses()
    
    setup_seed(script_args.random_seed)

    training_args.learning_rate *= (script_args.total_iteration-script_args.current_iteration+1)/script_args.total_iteration
    print("---------------")
    print(f"learning rate: {training_args.learning_rate}")
    print("---------------")
    
    downloaded = False
    print("---------------")
    print("begin to load SFT model")
    print("---------------")
    while not downloaded:
        try:
            model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, 
                                                 torch_dtype=torch.bfloat16,
                                                 use_flash_attention_2=True if script_args.use_flash_attention else None,
                                                 use_cache=False)
            tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path,use_cache=False)
            downloaded = True
        except Exception as error:
            print("An error occurred:", error)
            print("Failed to load the SFT model. Retrying....")
            time.sleep(2)

    print("---------------")
    print("Load the model successfully!")
    print("Now begin to process the training data.")
    print("---------------")
    with open(f"{script_args.dataset_path}/result.json",'r') as f:
        dataset = json.load(f)

    #with open("mistral_nvidia_math_raft_our_auto_regressive_select_n16_correct_subset.json") as f:
    #with open("mistral_nvidia_math_raft_vanilla_soft_auto_regressive_select_n16_correct_subset.json") as f:
        #correct_set = json.load(f)

    #dataset += correct_set
    #random.shuffle(dataset)
    
    #dataset = dataset['instances']
    # dataset_dict = {
    #     "text":[]
    # }
    # for sample in dataset:
    #     dataset_dict['text'].append(sample['text'])
    # dataset = Dataset.from_dict(dataset_dict)
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    # tokenizer.padding_side = "right"
    
    # finetuner = SFTTrainer(
    #     model=model,
    #     train_dataset=dataset,
    #     dataset_text_field="text",
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     max_seq_length = 512,
    # )

    dataset_dict = {
        "input_ids":[],
        "attention_mask":[],
        "labels":[]
    }
    for i,sample in tqdm(enumerate(dataset)):
        raw_text = sample['text']
        pattern = "Step 1:"
        idx = raw_text.find(pattern)-1
        prompt = raw_text[:idx]
        
        encode = tokenizer(raw_text,
                           add_special_tokens=True,
                            truncation=True)
        dataset_dict['input_ids'].append(encode['input_ids'])    
        dataset_dict['attention_mask'].append(encode['attention_mask'])
        labels = encode['input_ids'].copy()
        
        prompt = tokenizer(prompt,
                           add_special_tokens=True,
                            truncation=True)
        masked_length = len(prompt['input_ids'])
        labels[:masked_length] = [-100 for j in range(masked_length)]
        dataset_dict['labels'].append(labels)
        
        max_length = 512
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
    dataset = Dataset.from_dict(dataset_dict)
    
    finetuner = Trainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        #data_collator=collator,
    )
    print("---------------")
    print("begin to fine-tune the model")
    print("---------------")
    finetuner.train()

    print("---------------")
    print("Finish training the generator model!")
    print("Now begin to save the model.")
    print("---------------")
    finetuner.model.save_pretrained(training_args.output_dir)
    finetuner.tokenizer.save_pretrained(training_args.output_dir)
    print("---------------")
    print("Saved the trained model successfully.")
    print("---------------")
