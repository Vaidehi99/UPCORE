from sklearn.ensemble import IsolationForest
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import torch
from statistics import mean as mean2
import random
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import chi2
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

class args:
    model_name = "../cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/"
    topic_path = "/nas-ssd2/vaidehi/projects/Composition/wiki/topics.json"
    file_path = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/llama31_clusters/{}.jsonl"
    outpath = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/llama31_clusters/hiddenstate_variance_bert_clusters_outliers_removed_0.05.json"
    suffix = "_subsampled"
    out_suffix = "_outliers_subsampled"
    # suffix = "_removed_outliers"
    # out_suffix = "_outliers"

pre, post =[], []
model = AutoModelForCausalLM.from_pretrained(args.model_name, output_hidden_states=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model.eval()

def get_stats(path="/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/hiddenstate_variance_clusters_outliers_removed.json"):
  x=json.load(open(path, "r"))
  cur = list(x.values())
  lens_before, lens_after, var_before, var_after = [], [], [], []
  for d in cur:
    k_all = list(d.keys())
    v_all = list(d.values())
    len1, len2 = k_all[0], k_all[1]
    var1, var2 = v_all[0], v_all[1]
    lens_before.append(int(len1))
    lens_after.append(int(len2))
    var_before.append(float(var1))
    var_after.append(float(var2))

  print(mean(lens_before), mean(lens_after), mean(var_before), mean(var_after))
  
def subtract_jsonl_files(args):
  topics = json.load(open(args.topic_path,"r"))[:6]
    # Load all JSON objects from file2 into a set
  topics = ["tv_cluster_{}_dense5_filt".format(i) for i in range(10)]
  for topic in topics:
    f2 = open(args.file_path.format(topic).replace(".jsonl", "{}.jsonl".format(args.suffix)), 'r').readlines()# as f2:
    # import pdb; pdb.set_trace();
    set_file2 = [json.loads(line.strip()) for line in f2]

    # Open file1 and write only the lines not in file2 to the output
    with open(args.file_path.format(topic), 'r') as f1, open(args.file_path.format(topic).replace(".jsonl", "{}.jsonl".format(args.out_suffix)), 'w') as outfile:
        for line in f1:
            json_object = json.loads(line)
            if json_object not in set_file2:
                json.dump(json_object, outfile)
                outfile.write('\n')
  
def subsample_cluster(args):
  # topics = json.load(open(args.topic_path,"r"))[:6]
  topics = ["tv_cluster_{}_dense5_filt".format(i) for i in range(10)]
  for topic in tqdm(topics):

    cur_topic_hid_states = []
    data = []
    idx = []
    with open(args.file_path.format(topic), 'r') as f:
      for line in f:
        # Parse each line as a JSON object
        json_obj = json.loads(line)
        data.append(json_obj)
    num_lines = len(open(args.file_path.format(topic).replace(".jsonl", "_removed_outliers.jsonl"), 'r').readlines())
    data = random.sample(data, num_lines)
    with open(args.file_path.format(topic).replace(".jsonl", "_subsampled.jsonl"), "w") as jsonl_file:
      for entry in data:
        jsonl_file.write(json.dumps(entry) + "\n") 






def remove_outliers(questions, hidden_states):
    """
    Remove outliers from the hidden states using Isolation Forest.
    """
    # questions, hidden_states = zip(*questions_and_hidden_states)
    hidden_states_matrix = torch.stack(hidden_states).squeeze(1).cpu().numpy()
    
    # Fit Isolation Forest
    # clf = IsolationForest(random_state=42, contamination=0.1)
    clf = IsolationForest(random_state=42, contamination=0.1)

    inliers = clf.fit_predict(hidden_states_matrix) > 0
    # decision_scores = clf.decision_function(hidden_states_matrix)
    # raw_scores = clf.score_samples(hidden_states_matrix)

    # import pdb; pdb.set_trace();

    
    # Filter inliers
    filtered_questions = [q for i, q in enumerate(questions) if inliers[i]]
    filtered_hidden_states = [h for i, h in enumerate(hidden_states) if inliers[i]]
    return filtered_questions, filtered_hidden_states



def remove_outliers_lof(questions, hidden_states, n_neighbors=20, contamination=0.1):
    """
    Remove outliers from the hidden states using Local Outlier Factor (LOF).
    
    Args:
        questions (list): List of questions.
        hidden_states (list of torch.Tensor): Corresponding hidden states.
        n_neighbors (int): Number of neighbors to use for LOF (default 20).
        contamination (float): Estimated proportion of outliers in the data (default 0.1).
        
    Returns:
        tuple: Filtered questions and hidden states.
    """
    hidden_states_matrix = torch.stack(hidden_states).squeeze(1).cpu().numpy()
    
    # Fit Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    inliers = lof.fit_predict(hidden_states_matrix) == 1  # +1 = inlier, -1 = outlier

    # Filter inliers
    filtered_questions = [q for i, q in enumerate(questions) if inliers[i]]
    filtered_hidden_states = [h for i, h in enumerate(hidden_states) if inliers[i]]

    return filtered_questions, filtered_hidden_states



def remove_outliers_svm(questions, hidden_states, nu=0.1, kernel="rbf"):
    """
    Remove outliers from the hidden states using One-Class SVM.

    Args:
        questions (list): List of questions.
        hidden_states (list of torch.Tensor): Corresponding hidden states.
        nu (float): An upper bound on the fraction of outliers (default 0.1).
        kernel (str): Kernel type for One-Class SVM (default "rbf").

    Returns:
        tuple: Filtered questions and hidden states.
    """
    hidden_states_matrix = torch.stack(hidden_states).squeeze(1).cpu().numpy()
    
    # Fit One-Class SVM
    clf = OneClassSVM(kernel=kernel, nu=nu)
    clf.fit(hidden_states_matrix)
    
    # Predict inliers (+1) and outliers (-1)
    inliers = clf.predict(hidden_states_matrix) == 1

    # Filter inliers
    filtered_questions = [q for i, q in enumerate(questions) if inliers[i]]
    filtered_hidden_states = [h for i, h in enumerate(hidden_states) if inliers[i]]

    return filtered_questions, filtered_hidden_states


def remove_outliers_pca(questions, hidden_states, n_components=2, threshold=0.50):
    """
    Remove outliers from the hidden states using PCA and Mahalanobis distance.
    
    Args:
        questions (list): List of questions.
        hidden_states (list of torch.Tensor): Corresponding hidden states.
        n_components (int): Number of principal components to retain.
        threshold (float): Chi-square threshold for outlier detection.
        
    Returns:
        tuple: Filtered questions and hidden states.
    """
    hidden_states_matrix = torch.stack(hidden_states).squeeze(1).cpu().numpy()
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(hidden_states_matrix)
    
    # Compute Mahalanobis distance
    mean = np.mean(transformed, axis=0)
    cov_inv = np.linalg.inv(np.cov(transformed, rowvar=False))
    
    def mahalanobis(x, mean, cov_inv):
        diff = x - mean
        return np.sqrt(diff @ cov_inv @ diff.T)
    
    distances = np.array([mahalanobis(x, mean, cov_inv) for x in transformed])
    
    # Determine threshold based on chi-square distribution
    chi_square_threshold = chi2.ppf(threshold, df=n_components)
    
    inliers = distances < np.sqrt(chi_square_threshold)

    # Filter inliers
    filtered_questions = [q for i, q in enumerate(questions) if inliers[i]]
    filtered_hidden_states = [h for i, h in enumerate(hidden_states) if inliers[i]]
    
    return filtered_questions, filtered_hidden_states


def get_var(args):
  # topics = json.load(open(args.topic_path,"r"))[:6]
  topics = ["bert_cluster_{}".format(i) for i in range(0,4,1)]

  var = {}
  for topic in tqdm(topics):

    cur_topic_hid_states = []
    data = []
    idx = []
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
        idx.append(m)
    filtered_questions, filtered_hidden_states = remove_outliers(idx, cur_topic_hid_states)
    with open(args.file_path.format(topic).replace(".jsonl", "_removed_outliers_0.4.jsonl"), 'w') as file:
      for entry in filtered_questions:
        file.write(json.dumps(entry) + '\n')
    # pre, post = [], []
    # import pdb; pdb.set_trace();
    hidden_states_tensor = torch.stack(filtered_hidden_states)
    # import pdb; pdb.set_trace();
    mean = torch.mean(hidden_states_tensor, dim=0)
   
    centered_data = hidden_states_tensor - mean
    mean = mean.squeeze(0)
    centered_data = centered_data.squeeze(1)
    covariance_matrix = (centered_data.T @ centered_data) / (centered_data.shape[0] - 1)
    total_variance = torch.trace(covariance_matrix)
    print("Total Variance (Trace of Covariance Matrix) {}:".format(topic), total_variance.item())
    post.append(total_variance.item())

    hidden_states_tensor = torch.stack(cur_topic_hid_states)
    # import pdb; pdb.set_trace();
    mean = torch.mean(hidden_states_tensor, dim=0)
   
    centered_data = hidden_states_tensor - mean
    mean = mean.squeeze(0)
    centered_data = centered_data.squeeze(1)
    covariance_matrix = (centered_data.T @ centered_data) / (centered_data.shape[0] - 1)
    total_variance_orig = torch.trace(covariance_matrix)
    print("Total Variance (Trace of Covariance Matrix) {}:".format(topic), total_variance_orig.item())
    var[topic] = {len(filtered_questions): total_variance.item(), len(idx):total_variance_orig.item()}
    pre.append(total_variance_orig.item())
  # import pdb; pdb.set_trace();
  print(mean2(pre), mean2(post))
  json.dump(var, open(args.outpath, "w"))

def get_var_svm(args):
  # topics = json.load(open(args.topic_path,"r"))[:6]
  topics = ["bert_cluster_{}".format(i) for i in range(0,6,1)]

  var = {}
  for topic in tqdm(topics):

    cur_topic_hid_states = []
    data = []
    idx = []
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
        idx.append(m)
    filtered_questions, filtered_hidden_states = remove_outliers_svm(idx, cur_topic_hid_states)
    with open(args.file_path.format(topic).replace(".jsonl", "_removed_outliers_svm.jsonl"), 'w') as file:
      for entry in filtered_questions:
        file.write(json.dumps(entry) + '\n')
    # pre, post = [], []
    # import pdb; pdb.set_trace();
    hidden_states_tensor = torch.stack(filtered_hidden_states)
    # import pdb; pdb.set_trace();
    mean = torch.mean(hidden_states_tensor, dim=0)
   
    centered_data = hidden_states_tensor - mean
    mean = mean.squeeze(0)
    centered_data = centered_data.squeeze(1)
    covariance_matrix = (centered_data.T @ centered_data) / (centered_data.shape[0] - 1)
    total_variance = torch.trace(covariance_matrix)
    print("Total Variance (Trace of Covariance Matrix) {}:".format(topic), total_variance.item())
    post.append(total_variance.item())

    hidden_states_tensor = torch.stack(cur_topic_hid_states)
    # import pdb; pdb.set_trace();
    mean = torch.mean(hidden_states_tensor, dim=0)
   
    centered_data = hidden_states_tensor - mean
    mean = mean.squeeze(0)
    centered_data = centered_data.squeeze(1)
    covariance_matrix = (centered_data.T @ centered_data) / (centered_data.shape[0] - 1)
    total_variance_orig = torch.trace(covariance_matrix)
    print("Total Variance (Trace of Covariance Matrix) {}:".format(topic), total_variance_orig.item())
    var[topic] = {len(filtered_questions): total_variance.item(), len(idx):total_variance_orig.item()}
    pre.append(total_variance_orig.item())
  # import pdb; pdb.set_trace();
  print(mean2(pre), mean2(post))
  json.dump(var, open(args.outpath, "w"))


def remove_outliers_pca(questions, hidden_states, n_components=2, threshold=0.50):
    """
    Remove outliers from the hidden states using PCA and Mahalanobis distance.
    
    Args:
        questions (list): List of questions.
        hidden_states (list of torch.Tensor): Corresponding hidden states.
        n_components (int): Number of principal components to retain.
        threshold (float): Chi-square threshold for outlier detection.
        
    Returns:
        tuple: Filtered questions and hidden states.
    """
    hidden_states_matrix = torch.stack(hidden_states).squeeze(1).cpu().numpy()
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(hidden_states_matrix)
    
    # Compute Mahalanobis distance
    mean = np.mean(transformed, axis=0)
    cov_inv = np.linalg.inv(np.cov(transformed, rowvar=False))
    
    def mahalanobis(x, mean, cov_inv):
        diff = x - mean
        return np.sqrt(diff @ cov_inv @ diff.T)
    
    distances = np.array([mahalanobis(x, mean, cov_inv) for x in transformed])
    
    # Determine threshold based on chi-square distribution
    chi_square_threshold = chi2.ppf(threshold, df=n_components)
    
    inliers = distances < np.sqrt(chi_square_threshold)

    # Filter inliers
    filtered_questions = [q for i, q in enumerate(questions) if inliers[i]]
    filtered_hidden_states = [h for i, h in enumerate(hidden_states) if inliers[i]]
    
    return filtered_questions, filtered_hidden_states


def get_var(args):
  # topics = json.load(open(args.topic_path,"r"))[:6]
  topics = ["tv_cluster_{}_dense5_filt".format(i) for i in range(0,10,1)]

  var = {}
  for topic in tqdm(topics):

    cur_topic_hid_states = []
    data = []
    idx = []
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
        idx.append(m)
    filtered_questions, filtered_hidden_states = remove_outliers(idx, cur_topic_hid_states)
    with open(args.file_path.format(topic).replace(".jsonl", "_removed_outliers.jsonl"), 'w') as file:
      for entry in filtered_questions:
        file.write(json.dumps(entry) + '\n')
    # pre, post = [], []
    # import pdb; pdb.set_trace();
    hidden_states_tensor = torch.stack(filtered_hidden_states)
    # import pdb; pdb.set_trace();
    mean = torch.mean(hidden_states_tensor, dim=0)
   
    centered_data = hidden_states_tensor - mean
    mean = mean.squeeze(0)
    centered_data = centered_data.squeeze(1)
    covariance_matrix = (centered_data.T @ centered_data) / (centered_data.shape[0] - 1)
    total_variance = torch.trace(covariance_matrix)
    print("Total Variance (Trace of Covariance Matrix) {}:".format(topic), total_variance.item())
    post.append(total_variance.item())

    hidden_states_tensor = torch.stack(cur_topic_hid_states)
    # import pdb; pdb.set_trace();
    mean = torch.mean(hidden_states_tensor, dim=0)
   
    centered_data = hidden_states_tensor - mean
    mean = mean.squeeze(0)
    centered_data = centered_data.squeeze(1)
    covariance_matrix = (centered_data.T @ centered_data) / (centered_data.shape[0] - 1)
    total_variance_orig = torch.trace(covariance_matrix)
    print("Total Variance (Trace of Covariance Matrix) {}:".format(topic), total_variance_orig.item())
    var[topic] = {len(filtered_questions): total_variance.item(), len(idx):total_variance_orig.item()}
    pre.append(total_variance_orig.item())
  # import pdb; pdb.set_trace();
  print(mean2(pre), mean2(post))
  json.dump(var, open(args.outpath, "w"))

def get_var_lof(args):
  # topics = json.load(open(args.topic_path,"r"))[:6]
  topics = ["bert_cluster_{}".format(i) for i in range(0,6,1)]

  var = {}
  for topic in tqdm(topics):

    cur_topic_hid_states = []
    data = []
    idx = []
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
        idx.append(m)
    filtered_questions, filtered_hidden_states = remove_outliers_lof(idx, cur_topic_hid_states)
    with open(args.file_path.format(topic).replace(".jsonl", "_removed_outliers_lof.jsonl"), 'w') as file:
      for entry in filtered_questions:
        file.write(json.dumps(entry) + '\n')
    # pre, post = [], []
    # import pdb; pdb.set_trace();
    hidden_states_tensor = torch.stack(filtered_hidden_states)
    # import pdb; pdb.set_trace();
    mean = torch.mean(hidden_states_tensor, dim=0)
   
    centered_data = hidden_states_tensor - mean
    mean = mean.squeeze(0)
    centered_data = centered_data.squeeze(1)
    covariance_matrix = (centered_data.T @ centered_data) / (centered_data.shape[0] - 1)
    total_variance = torch.trace(covariance_matrix)
    print("Total Variance (Trace of Covariance Matrix) {}:".format(topic), total_variance.item())
    post.append(total_variance.item())

    hidden_states_tensor = torch.stack(cur_topic_hid_states)
    # import pdb; pdb.set_trace();
    mean = torch.mean(hidden_states_tensor, dim=0)
   
    centered_data = hidden_states_tensor - mean
    mean = mean.squeeze(0)
    centered_data = centered_data.squeeze(1)
    covariance_matrix = (centered_data.T @ centered_data) / (centered_data.shape[0] - 1)
    total_variance_orig = torch.trace(covariance_matrix)
    print("Total Variance (Trace of Covariance Matrix) {}:".format(topic), total_variance_orig.item())
    var[topic] = {len(filtered_questions): total_variance.item(), len(idx):total_variance_orig.item()}
    pre.append(total_variance_orig.item())
  # import pdb; pdb.set_trace();
  print(mean2(pre), mean2(post))
  json.dump(var, open(args.outpath, "w"))


if __name__ == "__main__":
  # subsample_cluster(args)
  # get_var(args)
  # get_stats(path="/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/hiddenstate_variance_min_var_clusters_outliers_removed.json")
  subtract_jsonl_files(args)
  # get_var_svm(args)
  # get_var_lof(args)
