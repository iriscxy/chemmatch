import os
import json

# Define the function to filter chemistry-related records
import pdb


def is_chemistry(fields_of_study):
    if not fields_of_study:
        return False
    return any(field['category'].lower() == 'chemistry' for field in fields_of_study)


# Get current directory
current_directory = os.getcwd()

# List all files in the current directory
files = os.listdir(current_directory)

# Filter for files that match the pattern papers-part*.jsonl
jsonl_files = [file for file in files if file.startswith('papers-part') and file.endswith('.jsonl')]

# Output file to store all filtered records
output_file_path = os.path.join(current_directory, "filtered_papers.jsonl")

# Process each file
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for file_name in jsonl_files:
        input_file_path = os.path.join(current_directory, file_name)

        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for line in infile:

                record = json.loads(line.strip())
                if is_chemistry(record.get('s2fieldsofstudy', [])):
                    outfile.write(line )

