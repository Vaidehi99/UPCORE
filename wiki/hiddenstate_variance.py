import torch


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm

class args:
    model_name = "../cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/"
    topic_path = "/nas-ssd2/vaidehi/projects/Composition/wiki/topics.json"
    file_path = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/llama31_clusters/{}.jsonl"
    outpath = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/hiddenstate_variance_clusters_coderate.json"

model = AutoModelForCausalLM.from_pretrained(args.model_name, output_hidden_states=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model.eval()


def get_var(args):
  topics = json.load(open(args.topic_path,"r"))
  topics = ["bert_cluster_{}".format(i) for i in range(6)]
  var = {}
  for topic in tqdm(topics):

    cur_topic_hid_states = []
    data = []
    with open(args.file_path.format(topic), 'r') as f:
      for line in f:
        # Parse each line as a JSON object
        json_obj = json.loads(line)
        data.append(json_obj)
    for m in tqdm(data):
        text = m['question']
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs)
        all_hidden_states = outputs.hidden_states 
        cur_topic_hid_states.append(all_hidden_states[-2][:, -1, :])
    hidden_states_tensor = torch.stack(cur_topic_hid_states)
    # torch.save(hidden_states_tensor, '/nas-ssd2/vaidehi/projects/Composition/wiki/hsv3.pt')
 
    mean = torch.mean(hidden_states_tensor, dim=0)
   
    centered_data = hidden_states_tensor - mean
    mean = mean.squeeze(0)
    centered_data = centered_data.squeeze(1)
    covariance_matrix = (centered_data.T @ centered_data) / (centered_data.shape[0] - 1)
    total_variance = torch.trace(covariance_matrix)
    print("Total Variance (Trace of Covariance Matrix) {}:".format(topic), total_variance.item())
    var[topic] = total_variance.item()
  json.dump(var, open(args.outpath, "w"))

if __name__ == "__main__":
  get_var(args)


