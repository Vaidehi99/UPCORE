import json, random
import numpy as np
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


class args:
    prob_path = "../cache/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/eval_results/ds_sizeFalse/eval_log_aggregated.json"
    data_path = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/counterfact400_filt.jsonl"
    input_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/counterfact5200.jsonl" 
    output_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/counterfact5200_filt.jsonl" 
    output_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/counterfact_merged_single_filt.jsonl"
    prob_path = "../cache/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/eval_results/ds_sizeFalse/eval_log_aggregated.json"
    data_path = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/counterfact_merged_single.jsonl"
    input_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/counterfact_merged_single_filt_cluster_bert.jsonl"
    output_dir = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/llama31_clusters/"

    input_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/counterfact_merged_single_top6_clusters.jsonl"    
    output_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/counterfact_merged_single_filt_cluster_min_var.jsonl"
    output_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/counterfact_merged_single_top6_clusters_cluster_min_var.jsonl"
    output_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/counterfact_merged_single_top6_clusters_cluster_min_hsv.jsonl"
    output_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/counterfact_merged_single_top6_clusters_cluster_random.jsonl"
    input_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/counterfact_merged_single_top6_clusters_cluster_random.jsonl"
    input_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/counterfact_merged_single_top6_clusters.jsonl" 
    input_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/counterfact_merged_single_top6_clusters_cluster_coderate.jsonl"
    input_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/bert_single_top6_clusters.jsonl"
    input_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/bert_single_top6_clusters_random_cluster.jsonl"
    input_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/bert_single_top6_clusters_nneigh_random_cluster.jsonl"
    output_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/bert_single_top6_clusters_nneigh_random_cluster.jsonl"


    # input_file = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/counterfact_merged_single_top6_clusters.jsonl"


    model_name = "../cache/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/"



def dump_prob(args):
    x = json.load(open(args.prob_path, "r"))
    data = []
    with open(args.data_path, 'r') as f:
        for line in f:
            # Parse each line as a JSON object and add it to the list
            data.append(json.loads(line))
    all_probs = np.exp(-1*np.array(list(x['eval_log_forget.json']['avg_gt_loss'].values())))
    assert(len(all_probs)==len(data))
    print(np.mean(all_probs))
    for i in range(len(data)):
        data[i]['prob'] = all_probs[i]
    with open(args.data_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def dump_random(args):
    data = []
    with open(args.data_path, 'r') as f:
        for line in f:
            # Parse each line as a JSON object and add it to the list
            data.append(json.loads(line))
    
    for i in range(len(data)):
        data[i]['random'] = random.choice([0.0, 1.0])

    with open(args.data_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def dump_gen_acc(args):
    x = json.load(open(args.prob_path, "r"))
    data = []
    with open(args.data_path, 'r') as f:
        for line in f:
            # Parse each line as a JSON object and add it to the list
            data.append(json.loads(line))
    all_probs = np.array(list(x['eval_log_forget.json']['gen_acc'].values()))
    print(len(all_probs))
    assert(len(all_probs)==len(data))
    print(np.mean(all_probs))
    for i in range(len(data)):
        data[i]['gen_acc'] = all_probs[i]
    with open(args.data_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def idx_filt(args):

# Initialize a list to hold the first 400 records
  subset = []

# Reading the first 400 lines from the input JSONL file
  with open(args.input_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        # import pdb; pdb.set_trace();
        if i < 400: # and i < 6000:  # Only process the first 400 lines
            subset.append(json.loads(line))
        # else:
        #     break

# Writing the subset to a new JSONL file
  with open(args.output_file, "w", encoding="utf-8") as f:
    for record in subset:
        f.write(json.dumps(record) + "\n")  # Convert each record to JSON and write it as a line

  print(f"First 400 lines have been written to {args.output_file}")


# Define the filtering condition
def filter_condition(record):
    return record['gen_acc'] == 1.0

# Read, filter, and write the JSONL file
def filt(args):
  with open(args.input_file, "r") as infile, open(args.output_file, "w") as outfile:
    for line in infile:
        record = json.loads(line)  # Parse JSON line
        if filter_condition(record):  # Apply filter condition
            outfile.write(json.dumps(record) + "\n")  # Write filtered line

import os

def split_jsonl(input_file, output_dir, lines_per_file=400):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r', encoding='utf-8') as infile:
        file_count = 0
        current_lines = []
        for i, line in enumerate(infile):
            current_lines.append(line)
            if (i + 1) % lines_per_file == 0:
                output_file = os.path.join(output_dir, f'min_var_cluster_{file_count:03d}.jsonl')
                with open(output_file, 'w', encoding='utf-8') as outfile:
                    outfile.writelines(current_lines)
                current_lines = []
                file_count += 1

        # Write any remaining lines
        if current_lines:
            output_file = os.path.join(output_dir, f'min_var_cluster_{file_count:03d}.jsonl')
            with open(output_file, 'w', encoding='utf-8') as outfile:
                outfile.writelines(current_lines)

    print(f"Split completed. Files are saved in '{output_dir}'.")


import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load JSONL file
def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

# Save JSONL file
def save_jsonl(data, output_path):
    with open(output_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + "\n")


def cluster_jsonl_with_tfidf(input_path, output_path, key, num_clusters=7):
    # Load data
    data = load_jsonl(input_path)
    
    # Extract values for clustering
    values = [item[key] for item in data]
    
    # Generate TF-IDF embeddings
    vectorizer = TfidfVectorizer(stop_words='english')
    embeddings = vectorizer.fit_transform(values)
    
    # Scale embeddings (optional, improves clustering performance)
    scaler = StandardScaler(with_mean=False)
    embeddings_scaled = scaler.fit_transform(embeddings.toarray())
    
    # Perform clustering using cosine similarity metric
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings_scaled)
    
    # Add cluster labels to data
    for i, item in enumerate(data):
        item['cluster'] = int(labels[i])
    
    # Save the updated data
    save_jsonl(data, output_path)

# Main clustering function
def cluster_jsonl_with_bert(input_path, output_path, key, num_clusters=5):
    # Load data
    data = load_jsonl(input_path)
    
    # Extract values for clustering
    values = [item[key] for item in data]
    
    # Generate BERT embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose an appropriate model
    embeddings = model.encode(values, convert_to_tensor=True)
    
    # Scale embeddings (optional, improves clustering performance)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings.cpu())
    
    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings_scaled)
    
    # Add cluster labels to data
    for i, item in enumerate(data):
        item['cluster'] = int(labels[i])
    
    # Save the updated data
    save_jsonl(data, output_path)

def split_jsonl_by_key(input_file, output_dir, key):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    clusters = defaultdict(list)

    # Read the JSONL file and group by the key
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line.strip())
            if key in data:
                clusters[data[key]].append(line)
            else:
                print(f"Warning: Key '{key}' not found in line: {line.strip()}")

    # Write each cluster to its own file
    for cluster_key, lines in clusters.items():
        safe_cluster_key = str(cluster_key).replace('/', '_').replace('\\', '_')  # Make filename safe
        output_file = os.path.join(output_dir, f'randombert_cluster_{safe_cluster_key}_nneigh.jsonl')
        print(output_file)
        # import pdb; pdb.set_trace();
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.writelines(lines)

    print(f"Split completed. Files are saved in '{output_dir}'.")

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Load JSONL file
def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

# Save JSONL file
def save_jsonl(data, output_path):
    with open(output_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + "\n")

# Function to compute variance of embeddings in a cluster
def compute_cluster_variance(embeddings, labels):
    variances = []
    for cluster_id in np.unique(labels):
        cluster_embeddings = embeddings[labels == cluster_id]
        variances.append(np.var(cluster_embeddings, axis=0).mean())
    return np.mean(variances)

# Main clustering function
def cluster_minimize_variance(input_path, output_path, key, num_clusters=7):
    # Load data
    data = load_jsonl(input_path)
    
    # Extract values for clustering
    values = [item[key] for item in data]
    
    # Generate BERT embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
    embeddings = np.array(model.encode(values))
    
    # Perform clustering with KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    # Minimize variance
    current_variance = compute_cluster_variance(embeddings, labels)
    print(f"Initial Cluster Variance: {current_variance:.4f}")
    
    # Iterative adjustment to minimize variance
    for _ in range(10):  # Number of iterations (tunable)
        centroids = kmeans.cluster_centers_
        closest_points, _ = pairwise_distances_argmin_min(centroids, embeddings)
        new_labels = KMeans(n_clusters=num_clusters, init=embeddings[closest_points]).fit_predict(embeddings)
        new_variance = compute_cluster_variance(embeddings, new_labels)
        if new_variance < current_variance:
            labels = new_labels
            current_variance = new_variance
        else:
            break
    
    print(f"Final Cluster Variance: {current_variance:.4f}")
    
    # Add cluster labels to data
    for i, item in enumerate(data):
        item['cluster'] = int(labels[i])
    
    # Save the updated data
    save_jsonl(data, output_path)

def merge_dicts_from_jsonl(file_list, output_file):
    merged_data = []
    
    # Loop through each file in the file list
    for file_name in file_list:
        with open(file_name, 'r') as infile:
            for line in infile:
                # Parse each line as a dictionary and append to the merged data list
                merged_data.append(json.loads(line))
    
    # Write the merged data into the output file, one dict per line
    with open(output_file, 'w') as outfile:
        for item in merged_data:
            json.dump(item, outfile)
            outfile.write("\n")


# Example usage


# # Usage
# input_file = 'your_file.jsonl'  # Path to your input JSONL file
# output_dir = 'output_parts'     # Directory to save the output files
# lines_per_file = 400            # Number of lines per file


import json
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Function to calculate the variance of hidden states
def get_var(hidden_states_tensor):
    mean = torch.mean(hidden_states_tensor, dim=0)
    centered_data = hidden_states_tensor - mean
    centered_data = centered_data.squeeze(1)
    covariance_matrix = (centered_data.T @ centered_data) / (centered_data.shape[0] - 1)
    total_variance = torch.trace(covariance_matrix)
    return total_variance.item()

# Function to load data from JSONL
def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

# Function to save JSONL data
def save_jsonl(data, output_path):
    with open(output_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + "\n")

# Cluster data to minimize hidden state variance
def cluster_minimize_variance(input_path, output_path, model, tokenizer, num_clusters=5):
    # Load data
    data = load_jsonl(input_path)
    
    # Extract the text for which we will compute embeddings and hidden states
    texts = [item['question'] for item in data]

    # Extract hidden states for all texts
    hidden_states = []
    for text in tqdm(texts):
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        all_hidden_states = outputs.hidden_states
        # Extract the second-to-last hidden state
        hidden_states.append(all_hidden_states[-2][:, -1, :].cpu())
    
    hidden_states_tensor = torch.stack(hidden_states)

    # Perform initial clustering with KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(hidden_states_tensor.numpy())

    # Function to compute variance of each cluster
    def compute_cluster_variance(hidden_states_tensor, labels):
        variances = []
        for cluster_id in np.unique(labels):
            cluster_embeddings = hidden_states_tensor[labels == cluster_id]
            variances.append(get_var(cluster_embeddings))
        return np.mean(variances)

    # Initial cluster variance
    current_variance = compute_cluster_variance(hidden_states_tensor, labels)
    print(f"Initial Cluster Variance: {current_variance:.4f}")

    # Iterative adjustment to minimize variance
    for _ in range(10):  # Number of iterations (tunable)
        centroids = kmeans.cluster_centers_
        closest_points, _ = pairwise_distances_argmin_min(centroids, hidden_states_tensor.numpy())
        new_labels = KMeans(n_clusters=num_clusters, init=hidden_states_tensor.numpy()[closest_points]).fit_predict(hidden_states_tensor.numpy())
        new_variance = compute_cluster_variance(hidden_states_tensor, new_labels)
        if new_variance < current_variance:
            labels = new_labels
            current_variance = new_variance
        else:
            break

    print(f"Final Cluster Variance: {current_variance:.4f}")

    # Add cluster labels to the data
    for i, item in enumerate(data):
        item['cluster'] = int(labels[i])

    # Save the updated data
    save_jsonl(data, output_path)

def save_clusters_to_jsonl(cluster_labels, file_path):
    """
    Save cluster labels to a JSONL file.

    Parameters:
        cluster_labels (ndarray): Cluster labels for each sample.
        file_path (str): Path to save the JSONL file.
    """
    with open(file_path, 'w') as f:
        for idx, label in enumerate(cluster_labels):
            json.dump({"id": idx, "cluster": int(label)}, f)
            f.write("\n")

def randomly_cluster_jsonl(input_file, output_file, num_clusters=6, cluster_key="cluster"):
    """
    Randomly clusters a JSONL file into `num_clusters`.

    Args:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path to save the output JSONL file.
        num_clusters (int): Number of clusters to create.
        cluster_key (str): Key to use for cluster assignment in the output.
    """
    clustered_data = []

    # Read and process the JSONL file
    with open(input_file, 'r') as infile:
        for line in infile:
            data = json.loads(line.strip())
            # Assign a random cluster
            data[cluster_key] = random.randint(0, num_clusters - 1)
            clustered_data.append(data)

    # Write clustered data to output JSONL file
    with open(output_file, 'w') as outfile:
        for item in clustered_data:
            outfile.write(json.dumps(item) + '\n')


# model = AutoModelForCausalLM.from_pretrained(args.model_name, output_hidden_states=True).to("cuda")
# tokenizer = AutoTokenizer.from_pretrained(args.model_name)
# model.eval()



import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

def coding_rate_clustering(embeddings, num_clusters=3, max_iter=100, tol=1e-4):
    """
    Perform clustering to minimize the coding rate.

    Parameters:
        embeddings (ndarray): High-dimensional data embeddings (n_samples, n_features).
        num_clusters (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.

    Returns:
        tuple:
            cluster_labels (ndarray): Cluster labels for each sample.
            coding_rate (float): Final coding rate.
    """
    n_samples = embeddings.shape[0]

    # Initialize clusters with KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, max_iter=10, n_init=1)
    cluster_labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    def compute_coding_rate(embeddings, labels, centroids):
        """Compute the coding rate for the current clustering."""
        # Calculate intra-cluster distances
        intra_distances = np.sum([
            np.sum(np.linalg.norm(embeddings[labels == i] - centroids[i], axis=1)**2)
            for i in range(num_clusters)
        ])

        # Calculate inter-cluster distances
        inter_distances = np.sum(cdist(centroids, centroids, 'sqeuclidean'))

        # Coding rate is proportional to intra-cluster distances and inversely proportional to inter-cluster distances
        return intra_distances / (inter_distances + 1e-8)

    # Iteratively minimize coding rate
    prev_coding_rate = float('inf')
    for iteration in range(max_iter):
        # Assign clusters based on nearest centroid
        distances = pairwise_distances(embeddings, centroids, metric='euclidean')
        cluster_labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([
            embeddings[cluster_labels == i].mean(axis=0) if np.any(cluster_labels == i) else centroids[i]
            for i in range(num_clusters)
        ])

        # Compute coding rate
        current_coding_rate = compute_coding_rate(embeddings, cluster_labels, new_centroids)

        # Check convergence
        if np.abs(prev_coding_rate - current_coding_rate) < tol:
            break

        centroids = new_centroids
        prev_coding_rate = current_coding_rate

    return cluster_labels, prev_coding_rate


def add_cluster_labels_to_jsonl(input_file_path, output_file_path, cluster_labels):
    """
    Add cluster labels to an existing JSONL file.

    Parameters:
        input_file_path (str): Path to the input JSONL file.
        output_file_path (str): Path to save the updated JSONL file.
        cluster_labels (list or ndarray): Cluster labels to add to the JSON objects.
    """
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for idx, line in enumerate(infile):
            data = json.loads(line)
            data['cluster'] = int(cluster_labels[idx])
            json.dump(data, outfile)
            outfile.write("\n")
'''
### coding rate
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    # # Generate sample data
    # embeddings, _ = make_blobs(n_samples=300, centers=4, n_features=10, random_state=42)
    # model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose an appropriate model
    # embeddings = model.encode(values, convert_to_tensor=True)
    data = load_jsonl(args.input_file)
    
    # Extract values for clustering
    values = [item['question'] for item in data]
    
    # Generate BERT embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose an appropriate model
    embeddings = model.encode(values, convert_to_tensor=True)

    # Perform clustering
    num_clusters = 7
    cluster_labels, final_coding_rate = coding_rate_clustering(embeddings.cpu(), num_clusters=num_clusters)
    from collections import Counter
    print("Cluster Labels:", cluster_labels)
    print(Counter(cluster_labels))
    print("Final Coding Rate:", final_coding_rate)
    add_cluster_labels_to_jsonl(args.input_file, args.output_file, cluster_labels)  

'''  

def get_neigh_data(cluster_path):
    data = load_jsonl(cluster_path)



#     # probs = 
if __name__ == "__main__":
    # dump_prob(args)
    # dump_gen_acc(args)
    # filt(args)
    # idx_filt(args)
    # dump_random(args)
    # split_jsonl(args.input_file, args.output_dir, lines_per_file=400)
    # cluster_jsonl_with_bert(args.input_file, args.output_file, "question", num_clusters=7)
    # split_jsonl_by_key(args.input_file, args.output_dir, "cluster")
    
    # input_files = ["/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/cluster_{}.jsonl".format(i) for i  in range(6)]  # Add your filenames here

    # merge_dicts_from_jsonl(input_files, args.output_file)
    # cluster_jsonl_with_bert(args.input_file, args.output_file, "question", num_clusters=7)
    # split_jsonl_by_key(args.input_file, args.output_dir, "cluster")
    # cluster_minimize_variance(args.input_file, args.output_file, model, tokenizer, num_clusters=7)
    # randomly_cluster_jsonl(args.input_file, args.output_file)
    # split_jsonl_by_key(args.input_file, args.output_dir, "cluster")
    # cluster_jsonl_with_tfidf(args.input_file, args.output_file, "question", num_clusters=7)
    # split_jsonl_by_key(args.input_file, args.output_dir, "cluster")
    # input_files = ["/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/llama31_clusters/bert_cluster_{}_nneigh.jsonl".format(i) for i  in range(6)]  # Add your filenames here

    # merge_dicts_from_jsonl(input_files, args.output_file)
    # randomly_cluster_jsonl(args.input_file, args.output_file)
    split_jsonl_by_key(args.input_file, args.output_dir, "cluster")

    

    




