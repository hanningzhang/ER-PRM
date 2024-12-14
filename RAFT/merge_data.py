# merge the selected rationales after reward scoring from multi-GPU to a single result.json file

import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default ="",help="the location of the splited data")
    return parser.parse_args()

args = parse_args()
file_list = os.listdir(args.output_dir)
name = file_list[0].split('_')[0]
merged_data = []

for file in file_list:
    with open(f"{args.output_dir}/{file}",'r') as f:
        merged_data += json.load(f)
        
with open(f"{args.output_dir}/{name}.json",'w') as f:
    json.dump(merged_data,f,indent=4,ensure_ascii=False)