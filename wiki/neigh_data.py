import os
import json
from openai import AzureOpenAI
# from dotenv import load_dotenv # pip install python-dotenv
import random


# data = json.load(open("lc_concepts.json", "r"))

def sample_indices(length, sample_size=10):
    """
    Randomly samples indices from a given length.
    
    Args:
        length (int): The total length of the range (0 to length-1).
        sample_size (int): The number of indices to sample (default is 10).
    
    Returns:
        List[int]: A list of randomly sampled indices.
    """
    if sample_size > length:
        raise ValueError("Sample size cannot be greater than the length.")
    
    return random.sample(range(length), sample_size)

def read_jsonl(file_path):
    """
    Reads a JSONL file and returns the data as a list of dictionaries.
    
    Args:
        file_path (str): The path to the JSONL file.
    
    Returns:
        List[dict]: A list of dictionaries representing the JSONL data.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse each line as JSON and append to the list
            data.append(json.loads(line.strip()))
    return data

def remove_repetitions_in_jsonl(file_path):
    unique_lines = set()

    # Read the existing file and collect unique lines
    with open(file_path, 'r') as f:
        for line in f:
            json_line = json.loads(line.strip())  # Parse the JSON line
            unique_lines.add(json.dumps(json_line, sort_keys=True))  # Sort keys for consistency

    # Write the unique lines back to the file
    with open(file_path, 'w') as f:
        for line in unique_lines:
            f.write(line + '\n')

    print(f"Repetitions removed and file saved successfully at {file_path}")


topics = json.load(open("/nas-ssd2/vaidehi/projects/Composition/wiki/topics.json", "r"))
topics = ["tv_cluster_{}_dense5_filt".format(i) for i in [0, 1, 2, 4, 5, 6, 8, 9]]
cur = topics[7]
file_path="/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/llama31_clusters/{}.jsonl".format(cur)
out_path = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/llama31_clusters/{}_nneigh.jsonl".format(cur)
data = read_jsonl(file_path)
idx = sample_indices(len(data), sample_size=10)


prompt = '''
Identify the topic based on these comma separated prompts: {}
Give 2 factual prompts that lie in the neighborhood of this topic but not within the topic such that they have one word answers along with a perturbed answer in jsonl format:
For example: {'question": "Kyoto University is located in the country of", "answer": "Japan", "perturbed_answer": ["India"]}. 
Directly generate the jsonl with 20 lines, do not give any other information in the response
'''
#Entertainment Trivia, specifically focusing on film, television, and celebrity culture 
# Nature, Botany, and Science Trivia
# Sports and Competitions Trivia,
#Music Trivia
#Literature and Theatre Trivia
#Mythology and Ancient History Trivia
#UK Politics, Government, and History
#Geography and U.S. Trivia
prompt = '''
Identify the topic based on these comma separated prompts: {}
Give 2 factual prompts that lie in the neighborhood of this topic of Geography and U.S. Trivia but NOT within the topic such that they have short answers along with a perturbed answer in jsonl format:
For example: {'question": "Kyoto University is located in the country of", "answer": "Japan", "perturbed_answer": ["India"]}. 
Directly generate the jsonl with 20 lines, do not give any other information in the response
'''

# prompt = '''
# Given the topic: {}
# Give diverse factual prompts that lie in the neighborhood of this topic but not within the topic such that they have one sentence answers along with a perturbed answer in jsonl format:
# For example: {'question": "Which country is Kyoto university located in?", "answer": "Kyoto university is located in Japan", "perturbed_answer": ["Kyoto university is located in India"]}. 
# Directly generate the jsonl with 20 lines, do not give any other information in the response
# '''
# prompt_new = prompt.replace("{}", cur)


ques = ",".join([data[m]['question'] for m in idx])
prompt_new = prompt.replace("{}", ques)



# import pdb; pdb.set_trace();

# remove_repetitions_in_jsonl(out_path)

all_lines = []
for i in range(120):
    # conc = data["concepts"][i]["subject_name"]
    # print(prompt_new)
    # import pdb; pdb.set_trace();
    
    client = AzureOpenAI(
  azure_endpoint = "", 
  api_key="",  
  api_version=""
)
    response = client.chat.completions.create(
      model="gpt-4o", # "deployment_name".
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_new},
     ],
      # messages=[
      #     {"role": "system", "content": "You are a helpful assistant."},
      #     {"role": "user", "content": "Give me five concepts which have multiple wikipedia pages with medium centrality and not so important in the wikidata graph. Please respond in json format with id and subject name as keys."},
      # ],
      response_format={ "type": "json_object" }
  )

    
    x = response.choices[0].message.content
    print(x)
    all_lines.append(json.loads(x.strip()))
    # data["concepts"][i]["doc_names"] = x
# with open(out_path, 'w') as f:
#     for line in all_lines:
#         json.dump(line, f)
#         f.write('\n')
with open(out_path, 'w') as f:
    for line in all_lines:
        json.dump(line, f)
        f.write('\n')

remove_repetitions_in_jsonl(out_path)
# json.dump(data, open("lc_concepts.json", "w"))

