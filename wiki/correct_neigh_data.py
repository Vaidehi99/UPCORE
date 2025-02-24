import json

def filter_jsonl(input_file, output_file, key):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                data = json.loads(line.strip())
                if 'perturbed_answer' in data:
                    value = data[key]
                else:
                    continue
                # Keep the line only if the value is a list with one string element
                if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
                    outfile.write(json.dumps(data) + '\n')
                # import pdb; pdb.set_trace();

            except:
                import pdb; pdb.set_trace()  # Skip lines with invalid JSON

# Example usage
input_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/llama31_clusters/tv_cluster_9_dense5_filt_nneigh.jsonl"   # Change to your actual file path
output_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/llama31_clusters/tv_cluster_9_dense5_filt_nneigh_filt.jsonl"
key_to_check = "perturbed_answer"

filter_jsonl(input_file, output_file, key_to_check)
