import gc
from typing import Optional, Union, List, Tuple
import argparse
import json
import re
import random
import stanza
import numpy as np
import accelerate
from tqdm import tqdm
from datasets import load_dataset, Dataset
from dataclasses import dataclass, field
import time

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    HfArgumentParser,
    Trainer,
    default_data_collator,
    LlamaForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(default='mistralai/Mistral-7B-v0.1')  # sft model
    use_flash_attention: bool = field(default=True) 
    dataset_path: str = field(default="peiyi9979/Math-Shepherd")
    block_size: int =field(default=1024)
    current_iteration: int = field(default=1)
    total_iteration: int = field(default=3)

class CustomLlamaForCausalLM(LlamaForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        values: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #return_dict = True

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # if labels is None and not is_torchdynamo_compiling():
            #     logger.warning_once(
            #         "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
            #     )
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            # TODO: remove the float() operation in v4.46
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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

    setup_seed(42)
    training_args.remove_unused_columns = False
    #training_args.learning_rate *= (script_args.total_iteration-script_args.current_iteration+1)/script_args.total_iteration
     
    downloaded = False
    print("---------------")
    print("begin to load the base model")
    print("---------------")
    while not downloaded:
        try:
            model = CustomLlamaForCausalLM.from_pretrained(script_args.model_name_or_path, 
                                                 torch_dtype=torch.bfloat16,
                                                 use_flash_attention_2=True if script_args.use_flash_attention else None)
            tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
            downloaded = True
        except Exception as error:
            print("An error occurred:", error)
            print("Failed to load the SFT model. Retrying....")
            time.sleep(2)
        
    with open(script_args.dataset_path,'r') as f:
        raw_datasets = json.load(f)[:1000]
    
    random.shuffle(raw_datasets) 
    step_tag = 'ки'
    step_tag_id = tokenizer.encode(f"{step_tag}")[-1] # 12902
    plus_tag_id = tokenizer.encode('+')[-1]
    minus_tag_id = tokenizer.encode('-')[-1]
    
    dataset_dict = {
        "input_ids":[],
        "attention_mask":[],
        "labels":[],
        "values":[]
        #"test":[]
    }
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    #tokenizer.padding_side = "left"
        
    print(tokenizer.pad_token)
    
    print("---------------")
    print("begin to process the dataset")
    print("---------------")
        
    # shift version
    count_invalid = 0
    #if "Llama-3" in script_args.model_name_or_path:
    if True:
        step_tag = 'ки'
        step_tag_id = tokenizer.encode(f" {step_tag}")[-1] 
        plus_tag_id = tokenizer.encode(' +')[-1]
        minus_tag_id = tokenizer.encode(' -')[-1]
        #print(plus_tag_id)
        #print(minus_tag_id)
        for i in tqdm(range(len(raw_datasets))):
        #for i in tqdm(range(1000)):
            value_list = [] #starting
            counter = 0
            raw_text = raw_datasets[i]['text']
            raw_text = raw_text.replace(" ки\n"," ки")
            encode = tokenizer(raw_text,
                           add_special_tokens=True,
                            truncation=True)
            #print(encode['input_ids'].copy())
            new_encode_id = encode['input_ids'].copy()
            new_encode_id.append(tokenizer.pad_token_id)
            #print(new_encode_id)
            labels = encode['input_ids'].copy()
            
            raw_label = raw_datasets[i]['text']
            count_label = raw_label.count("ки")
            # if len(raw_datasets[i]['value']) < count_label:
            #     raw_datasets[i]['value'] = raw_datasets[i]['value'][:count_label]
            raw_label = raw_label.replace(" ки\n"," ки")
            raw_label = raw_label.replace(" ки"," +")
            reference_labels = tokenizer(raw_label)['input_ids']
            reference_labels = [tokenizer.pad_token_id] + reference_labels
            if not len(reference_labels) == len(new_encode_id):
                count_invalid += 1
                continue
            dataset_dict['input_ids'].append(encode['input_ids'])
            dataset_dict['input_ids'][-1].extend([tokenizer.pad_token_id])
        
            dataset_dict['attention_mask'].append(encode['attention_mask'])
            dataset_dict['attention_mask'][-1].extend([0])
            for j in range(len(reference_labels)):
                if j == 0:
                    value_list.append(0)
                    reference_labels[0] = -100
                    continue
                if new_encode_id[j-1] == step_tag_id:
                    if counter < len(raw_datasets[i]['value']):
                        # if raw_datasets[i]['value'][counter] > 0:
                        #     value_list.append(1)
                        # else:
                        #     value_list.append(0)
                        value_list.append(raw_datasets[i]['value'][counter])
                        counter += 1
                        continue
                    else:
                        value_list.append(0)
                        reference_labels[j] = -100
                else:
                    reference_labels[j] = -100
                    value_list.append(0)
            dataset_dict['labels'].append(reference_labels)
            dataset_dict['values'].append(value_list)
            #print(value_list)
            #dataset_dict['test'].append(raw_datasets[i]['value'])
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
                if reference_labels[i] == plus_tag_id:
                    continue
                else:
                    reference_labels[i] = -100
            dataset_dict['labels'].append(reference_labels)
    
    print(f"invalid: {count_invalid}")
    for i in tqdm(range(len(dataset_dict['input_ids']))):
        block_size = script_args.block_size
        max_length = min(block_size, tokenizer.model_max_length)
        pad_length = max_length - len(dataset_dict["input_ids"][i])
        if pad_length < 0:
            # Truncates too long samples
            for key in ["input_ids", "attention_mask", "labels", "values"]:
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
            dataset_dict["values"][i].extend(
                [0 for _ in range(pad_length)]
            )
        #dataset_dict['test'].append(dataset_dict['labels'][i])
    
    #print(dataset_dict['input_ids'][0])                   
    dataset = Dataset.from_dict(dataset_dict)
    
    data_collator = default_data_collator
    
    class AutoRegressiveTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            #print(model.fc1.weight.dtype)
            #print(labels.shape)
            #print(inputs['values'].shape)
            #plus_id = 489
            #minus_id = 482
            plus_id = self.tokenizer.encode(' +')[-1]
            minus_id = self.tokenizer.encode(' -')[-1]
            #print(plus_id)
            values = inputs['values']
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]
            
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            values = values[..., 1:].contiguous()
            values = values.to(torch.bfloat16)

            plus_logits = logits[:,:,plus_id]
            minus_logits = logits[:,:,minus_id]
            #probs = torch.nn.functional.softmax(logits, dim=-1)
            #log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            if labels.dim() == plus_logits.dim() - 1:
                labels = labels.unsqueeze(-1)
                values = values.unsqueeze(-1)

            #padding_mask = labels.eq(-100)
            # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
            # will ignore them in any case.
            #labels = torch.clamp(labels, min=0)
            #logits_prob = probs.gather(dim=-1, index=labels)
            
            chosen = labels != -100
            
            pred_plus_values = plus_logits[chosen]
            pred_minus_values = minus_logits[chosen]
            gt_values = values[chosen]
            pred_combined = torch.stack((pred_plus_values, pred_minus_values), dim=1)
            #print(pred_combined.shape)
            gt_negative = 1 - gt_values
            gt_combined = torch.stack((gt_values, gt_negative), dim=1)
            # print(pred_minus_values.shape)
            # print(gt_values.shape)
            #logits_prob = logits_prob.to(torch.bfloat16)
            #print(logits_prob.dtype)
            #print(values.dtype)
            #logits_prob[labels == 0] = 0
            #print(logits_prob)
            #print(values)
            #loss = torch.nn.functional.mse_loss(logits_prob, values,reduction='sum')
            loss = torch.nn.functional.cross_entropy(pred_combined,gt_combined,size_average=True)
            #loss = values
            #loss.masked_fill_(padding_mask, 0.0)
            #num_active_elements = padding_mask.numel() - padding_mask.long().sum()
            #loss = loss / num_active_elements
            loss = loss.to(torch.bfloat16)
            #print(loss.dtype)
            #print(nll_loss.shape)
            # works for fp16 input tensor too, by internally upcasting it to fp32
            #smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

            #nll_loss.masked_fill_(padding_mask, 0.0)
            #smoothed_loss.masked_fill_(padding_mask, 0.0)

            # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
            # num_active_elements = padding_mask.numel() - padding_mask.long().sum()
            # nll_loss = nll_loss.sum() / num_active_elements
            #smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
            #loss = nll_loss
            if return_outputs:
                return loss, {"output": outputs}
            return loss
    
    finetuner = AutoRegressiveTrainer(
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
