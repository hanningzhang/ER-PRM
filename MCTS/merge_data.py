import os
import json

# Directory path where your files are located
directory_path = 'smooth_reward_data'

# Output file to store the combined data
output_file = 'combined_data.json'

# List to hold the contents from each file
combined_data = []

# Iterate over each file in the directory
for filename in sorted(os.listdir(directory_path)):
    file_path = os.path.join(directory_path, filename)
        
        # Open the file and read its content
    with open(file_path, 'r') as file:
        data = json.load(file)
        combined_data += data

# Write the combined data to the output file
with open(output_file, 'w',) as output:
    json.dump(combined_data,output,indent=4,ensure_ascii=False)

print(f"Data from files containing 'gsm' has been combined and saved in {output_file}.")
