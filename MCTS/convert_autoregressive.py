import json
from tqdm import tqdm

def process_text(instance):
    if len(instance) == 1:
        return instance[0]['text']
    text = instance[0]['text'].strip()
    for i,sample in enumerate(instance[:-1]):
        text_diff = instance[i+1]['text'][len(instance[i]['text']):]
        text_diff = text_diff.strip()
        text += " ки\n" + text_diff
        text = text.strip()
    return text
        
    
with open("score_data.json",'r') as f:
    label_data = json.load(f)
    
data = label_data
dataset = []
finish_idx = []
count_diff = 0
count = 0
length = 0

for i in tqdm(range(len(data))):
    if "The answer is" in data[i]['text']:
        finish_idx.append(i)
            
for i in tqdm(range(len(finish_idx))):
    if i == 0:
        instance = data[:finish_idx[i]+1]
    else:
        instance = data[finish_idx[i-1]+1:finish_idx[i]+1]
    value_list = []
    for v in instance:
        value_list.append(v['label'])
    #text = data[finish_idx[i]]['text'] + " ки"
    text = process_text(instance)
    if text.count("ки") != len(value_list):
        count_diff += 1
        
    # new_value_list = []
    # for j in value_list:
    #     if j > 0:
    #         new_value_list.append(1.0)
    #     else:
    #         new_value_list.append(0.0)
    # value_list = new_value_list
    dataset.append({"text":text,"value":value_list})
print(dataset[0])
print(count_diff)
with open("autoregressive_deepseek_data.json",'w') as f:
    json.dump(dataset,f,indent=4,ensure_ascii=False)