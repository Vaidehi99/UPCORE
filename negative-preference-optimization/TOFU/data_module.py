import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import os
import json
from utils import get_model_identifiers_from_yaml
single_token = True


concepts_list = ["Ancient_Egypt", "British_Empire", "China", "Cold_War", "French_Revolution", "New_York", "United_Nations", "USA", "World_War_1", "World_War_2"]
m = json.load(open("/nas-ssd2/vaidehi/projects/Composition/wiki/topics.json", "r"))[:6]
topics = m + [i+"_short" for i in m] + [i+"_nneigh_short" for i in m] + [i+"_nneigh" for i in m]+ [i+"_subsampled" for i in m] + [i+"_removed_outliers" for i in m] + [i+"_outliers" for i in m] + [i+"_outliers_subsampled" for i in m] +["bert_cluster_1_removed_outliers_0.05"] + ["neigh_samp", "Earth_question", "Ganzfeldeffect_question","Earth_jailbreak", "Ganzfeldeffect_jailbreak"] + ["counterfact400", "counterfact_retain", "counterfact5200", "counterfact400_filt", "counterfact_merged_single", "counterfact_llama31_correct"]+ ['bert_cluster_{}_nneigh'.format(i) for i in range(6)] + ['randombert_cluster_{}'.format(i) for i in range(6)]+ ['emb_ans_cluster_{}'.format(i) for i in range(7)]+ ['randombert_cluster_{}_subsampled'.format(i) for i in range(7)]+ ['bert_cluster_{}_outliers'.format(i) for i in range(7)]+ ['bert_cluster_{}_outliers_subsampled'.format(i) for i in range(7)]+ ['randombert_cluster_{}_removed_outliers'.format(i) for i in range(7)]+ ['emb_ans_cluster_{}_removed_outliers'.format(i) for i in range(7)]+ ['coderate_cluster_{}'.format(i) for i in range(7)] + ['bert_cluster_{}'.format(i) for i in range(7)] + ['coderate_cluster_{}_subsampled'.format(i) for i in range(7)] + ['min_var_cluster_{}'.format(i) for i in range(7)] + ['coderate_cluster_{}_removed_outliers'.format(i) for i in range(7)] + ['bert_cluster_{}_removed_outliers'.format(i) for i in range(7)]+ ['bert_cluster_{}_subsampled'.format(i) for i in range(7)] + ['min_var_cluster_{}_removed_outliers'.format(i) for i in range(7)] + ['cluster_{}_subsampled'.format(i) for i in range(7)] + ['min_var_cluster_{}_subsampled'.format(i) for i in range(7)] + ['random_cluster_{}'.format(i) for i in range(7)]

# topics = json.load(open("/nas-ssd2/vaidehi/projects/Composition/wiki/topics.json", "r")) + ["Earth_question", "Ganzfeldeffect_question","Earth_jailbreak", "Ganzfeldeffect_jailbreak"]+ ["counterfact400", "counterfact_retain"]
#0.19849666130438975#0.19341316483084797#0.1367385458393357#0.15465299910137564#0.17071765010458867#0.18062195539533601#0.11105532385890839#0.11033187643885135#0.14748746327627976#0.16495132408553276#0.11255432427307557 #0.19270387502684402 #0.11361826308020545 #0.1191
thresholds = {"Acidrain": 0.1191, "Aerosol":11361826308020545, "AlbertEinstein":0.19270387502684402, "Chindogu": 0.11255432427307557,
              "Chromium": 0.16495132408553276, "Earth": 0.14748746327627976, "Epoxy": 0.11033187643885135, "Ganzfeldeffect": 0.11105532385890839,
              "Hypercolor": 0.18062195539533601, "Kleinbottle":0.17071765010458867, "Pataphysics":0.15465299910137564, "Receiver": 0.1367385458393357,
              "UnitedStates": 0.19341316483084797, "WorldWarII": 0.19849666130438975, "Europe":0.19485963398857833}


th=thresholds["Ganzfeldeffect"]

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    if single_token:
        question_end_token = ""
        question_start_token = ""
        answer_token = " "
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    if single_token:
        num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=False))
    else:
        num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    if single_token:
        encoded = tokenizer(full_text, add_special_tokens=False, max_length=max_length, truncation=True,)

    else:
        encoded = tokenizer(
        full_text, 
        add_special_tokens=True,     
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        if single_token:
            label = encoded['input_ids'] + [-100] * (pad_length)
        else:
            label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)
    


class TextForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10", loss_type="idk"):
        super(TextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if split in topics:
            self.forget_data = datasets.load_dataset("json", data_files="/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/llama31_clusters/{}.jsonl".format(split.replace(" ", "_")), split="train")                                
        elif './TOFU_data' not in data_path: # load dataset from hugingface hub.
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
        elif split=="HC":
            self.forget_data = datasets.load_dataset("json", data_files="../wiki/hc_concepts_refusal_data_processed.jsonl", split="train")
        elif split=="MC":
            self.forget_data = datasets.load_dataset("json", data_files="../wiki/mc_concepts_refusal_data_processed.jsonl", split="train")
        elif split=="LC":
            self.forget_data = datasets.load_dataset("json", data_files="../wiki/lc_concepts_refusal_data_processed.jsonl", split="train")
        elif split in concepts_list:
            self.forget_data = datasets.load_dataset("json", data_files="../wiki/random_concepts/{}.jsonl".format(split), split="train")  
        elif split in topics:
            self.forget_data = datasets.load_dataset("json", data_files="/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/llama31_clusters/{}.jsonl".format(split.replace(" ", "_")), split="train")                                
        else:
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
        
        retain_split = "counterfact_retain"#"retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        if retain_split == "benign_dataset":
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        elif retain_split in topics:
            self.retain_data = datasets.load_dataset("json", data_files="/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/llama31_clusters/{}.jsonl".format(split.replace(" ", "_")), split="train")                                
        else:
            self.retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            
            torch.manual_seed(idx)
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[idx]['question']
            answer = data[idx]['answer']

            if data_type == "idk":
                #get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()
                
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets


class TextForgetDatasetDPOQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10", ):
        super(TextForgetDatasetDPOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if './TOFU_data' not in data_path:
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
        else:
            self.forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

        self.idontknowfile = "data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        retain_split = "counterfact_retain" #"retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        if retain_split == "benign_dataset":
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        elif retain_split in topics:
            self.retain_data = datasets.load_dataset("json", data_files="/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/llama31_clusters/{}.jsonl".format(split.replace(" ", "_")), split="train")                                
        else:
            self.retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:

            torch.manual_seed(idx)
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            
            question = data[idx]['question']
            
            if data_type != "idk":
                answer = data[idx]['answer']
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets


class TextForgetDatasetKTOQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10", ):
        super(TextForgetDatasetKTOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if './TOFU_data' not in data_path:
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
        else:
            self.forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

        self.idontknowfile = "data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        if retain_split == "benign_dataset":
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        elif retain_split in topics:
            self.retain_data = datasets.load_dataset("json", data_files="/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/llama31_clusters/{}.jsonl".format(split.replace(" ", "_")), split="train")                                
        else:
            self.retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:

            torch.manual_seed(idx)
            
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            
            question = data[idx]['question']
            
            if data_type != "idk":
                answer = data[idx]['answer']
                converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
                rets.append(converted_data)
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()

                answer = self.idk[rand_pos].strip()
                converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
                rets.append(converted_data)
        return rets


class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, question_key='question', answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if split=="HC":
            self.data = datasets.load_dataset("json", data_files="../wiki/hc_concepts_refusal_data_processed.jsonl", split="train")
        elif split=="MC":
            self.data = datasets.load_dataset("json", data_files="../wiki/mc_concepts_refusal_data_processed.jsonl", split="train")
        elif split=="LC":
            self.data = datasets.load_dataset("json", data_files="../wiki/lc_concepts_refusal_data_processed.jsonl", split="train")
        elif split in concepts_list:
            self.data = datasets.load_dataset("json", data_files="../wiki/random_concepts/{}.jsonl".format(split), split="train") 
        elif split in topics:
            self.data = datasets.load_dataset("json", data_files="/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/llama31_clusters/{}.jsonl".format(split.replace(" ", "_")), split="train")
        elif './TOFU_data' not in data_path: # load dataset from hugingface hub.
            self.data = datasets.load_dataset(data_path, split)["train"]
        else:
            self.data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze()


def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss
